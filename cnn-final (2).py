# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
# !pip -q install facenet-pytorch  # (magic command commented out)
import sys, subprocess
# !pip -q install timm facenet-pytorch  # (magic command commented out)

# %% [code] Cell 2
import os
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from facenet_pytorch import MTCNN
import glob
# !pip -q install tqdm  # (magic command commented out)
from tqdm.auto import tqdm
import random
from collections import defaultdict

import numpy as np


import matplotlib.pyplot as plt
import cv2

# !pip uninstall -y mxnet-cu112 mxnet-cu110 mxnet  # (magic command commented out)
# !pip install mxnet  # (magic command commented out)
# !pip -q install torchattacks timm  # (magic command commented out)

np.bool = np.bool_ 

import mxnet as mx 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import io
import copy


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# %% [code] Cell 3
os.environ["HF_HOME"] = "/kaggle/working/hf"

device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(image_size=112, margin=0, post_process=True, device=device)

cnn = timm.create_model(
    "hf_hub:gaunernst/convnext_nano.cosface_ms1mv3",
    pretrained=True
).to(device).eval()

# %% [code] Cell 4

print("Datasets under /kaggle/input:")
print(os.listdir("/kaggle/input")[:50])

imgs = glob.glob("/kaggle/input/**/*.*", recursive=True)
imgs = [p for p in imgs if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
print("num images found:", len(imgs))
print("first 20 images:")
print("\n".join(imgs[:20]))

# %% [code] Cell 5



os.environ["HF_HOME"] = "/kaggle/working/hf"  
device = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 64
MAX_IDENTITIES = 2000        
MAX_PROBES_PER_ID = 3        
GALLERY_PER_ID = 1           

def find_dataset_root():
    candidates = []
    for root in glob.glob("/kaggle/input/**", recursive=True):
        if not os.path.isdir(root): 
            continue
        subdirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
        if len(subdirs) == 0:
            continue
        img_count = 0
        for sd in subdirs[:10]:
            img_count += len(glob.glob(os.path.join(sd, "*.jpg")))
            img_count += len(glob.glob(os.path.join(sd, "*.png")))
            if img_count > 20:
                candidates.append(root)
                break
    if not candidates:
        raise RuntimeError("No dataset folders found under /kaggle/input. Did you add a dataset via 'Add data'?")
    candidates = sorted(set(candidates), key=lambda p: -p.count(os.sep))
    return candidates[0]

DATA_ROOT = find_dataset_root()
print("Using DATA_ROOT:", DATA_ROOT)



def build_id_map(root):
    id2paths = defaultdict(list)
    for person_dir in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(person_dir):
            continue
        person = os.path.basename(person_dir)
        imgs = []
        for ext in ("jpg", "jpeg", "png", "bmp", "webp"):
            imgs += glob.glob(os.path.join(person_dir, f"*.{ext}"))
        if len(imgs) >= 2:  
            id2paths[person].extend(sorted(imgs))
    return id2paths

id2paths = build_id_map(DATA_ROOT)
print("Identities with >=2 images:", len(id2paths))

ids = sorted(list(id2paths.keys()))
random.shuffle(ids)
ids = ids[:min(MAX_IDENTITIES, len(ids))]

gallery_paths = []
gallery_labels = []
probe_paths = []
probe_labels = []

label_map = {name: i for i, name in enumerate(ids)}

for name in ids:
    paths = id2paths[name]
    random.shuffle(paths)
    gallery_paths.append(paths[0])
    gallery_labels.append(label_map[name])

    probes = paths[1:1+MAX_PROBES_PER_ID]
    for p in probes:
        probe_paths.append(p)
        probe_labels.append(label_map[name])

print("Gallery:", len(gallery_paths), "Probes:", len(probe_paths))

mtcnn = MTCNN(image_size=112, margin=0, post_process=True, device=device)



REFERENCE_PTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.6963],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.3655]   # right mouth corner
], dtype=np.float32)

def align_face_warp(img, landmarks):
    if isinstance(img, Image.Image):
        img = np.array(img)
        
    src_pts = np.array(landmarks, dtype=np.float32)
    
    M, _ = cv2.estimateAffinePartial2D(src_pts, REFERENCE_PTS)
    
    warped = cv2.warpAffine(img, M, (112, 112))
    
    return Image.fromarray(warped)

def align_one(path):
    img = Image.open(path).convert("RGB")
    
    boxes, probs, points = mtcnn.detect(img, landmarks=True)
    
    if boxes is None:
        return None
        
    best_idx = np.argmax(probs)
    landmarks = points[best_idx] 
    
    face_aligned = align_face_warp(img, landmarks)
    
    face_tensor = torch.tensor(np.array(face_aligned)).permute(2,0,1).float()
    face_tensor = (face_tensor - 127.5) / 128.0 
    
    return face_tensor

def align_paths(paths, desc="align"):
    faces = []
    kept_paths = []
    for p in tqdm(paths, desc=desc, unit="img"):
        face = align_one(p)
        if face is not None:
            faces.append(face)
            kept_paths.append(p)
    return faces, kept_paths


gallery_faces, gallery_paths_kept = align_paths(gallery_paths, "align gallery")
probe_faces, probe_paths_kept = align_paths(probe_paths, "align probe")

gallery_labels_kept = [gallery_labels[gallery_paths.index(p)] for p in gallery_paths_kept]
probe_labels_kept   = [probe_labels[probe_paths.index(p)] for p in probe_paths_kept]

print("Kept gallery:", len(gallery_faces), "Kept probes:", len(probe_faces))

cnn = timm.create_model("hf_hub:gaunernst/convnext_nano.cosface_ms1mv3", pretrained=True).to(device).eval()

@torch.no_grad()
def embed(model, faces, batch_size=BATCH_SIZE, desc="embed"):
    embs = []
    for i in tqdm(range(0, len(faces), batch_size), desc=desc, unit="batch"):
        x = torch.stack(faces[i:i+batch_size]).to(device)  
        y = model(x)
        y = F.normalize(y, dim=1)
        embs.append(y.detach().cpu())
    return torch.cat(embs, dim=0)

g_cnn = embed(cnn, gallery_faces)
p_cnn = embed(cnn, probe_faces)


def identification_metrics(g_emb, g_labels, p_emb, p_labels, max_rank=20):
    sim = p_emb @ g_emb.T 
    g_labels_t = torch.tensor(g_labels)
    p_labels_t = torch.tensor(p_labels)

    topk = torch.topk(sim, k=min(max_rank, sim.shape[1]), dim=1).indices  
    topk_labels = g_labels_t[topk]  

    correct = (topk_labels == p_labels_t.unsqueeze(1))  
    rank1 = correct[:, 0].float().mean().item()
    rank5 = correct[:, :5].any(dim=1).float().mean().item() if sim.shape[1] >= 5 else float("nan")

    cmc = []
    for r in range(1, min(max_rank, sim.shape[1]) + 1):
        cmc.append(correct[:, :r].any(dim=1).float().mean().item())
    return rank1, rank5, cmc

rank1_cnn, rank5_cnn, cmc_cnn = identification_metrics(g_cnn, gallery_labels_kept, p_cnn, probe_labels_kept)

print(f"CNN  - Rank-1: {rank1_cnn:.4f} | Rank-5: {rank5_cnn:.4f}")

plt.figure()
plt.plot(range(1, len(cmc_cnn)+1), cmc_cnn, label="CNN")
plt.xlabel("Rank")
plt.ylabel("Identification Rate (CMC)")
plt.title("Closed-set Identification on LFW-style split")
plt.legend()
plt.show()


# %% [code] Cell 6

CONFIG = {
    'batch_size': 64,
    'lr': 1e-4,                 
    'epochs': 10,               
    'max_epsilon': 0.1,         
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'rec_path': '/kaggle/input/ms1mv3/ms1m-retinaface-t1' 
}

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(MXFaceDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.imgidx = np.array(range(1, int(header.label[1])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        while True:
            try:
                idx = self.imgidx[index]
                s = self.imgrec.read_idx(idx)
                header, img_bytes = mx.recordio.unpack(s)
                if len(img_bytes) == 0: raise ValueError
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != 'RGB': img = img.convert('RGB')
                if self.transform: img = self.transform(img)
                return img, 0 
            except:
                index = (index + 1) % len(self.imgidx)

    def __len__(self):
        return len(self.imgidx)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading Dataset...")
train_dataset = MXFaceDataset(root_dir=CONFIG['rec_path'], transform=transform)
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, drop_last=True)

model_name = "hf_hub:gaunernst/convnext_nano.cosface_ms1mv3"

print(f"Initializing Teacher ({model_name})...")
teacher_model = timm.create_model(model_name, pretrained=True).to(CONFIG['device'])
teacher_model.eval()
for param in teacher_model.parameters(): param.requires_grad = False

print(f"Initializing Student ({model_name})...")
student_model = timm.create_model(model_name, pretrained=True).to(CONFIG['device'])
student_model.train()

optimizer = optim.Adam(student_model.parameters(), lr=CONFIG['lr'])

def train_pgd_attack(model, images, target_embs, eps, alpha=2/255, steps=5):
    adv_images = images.clone().detach()
    for _ in range(steps):
        adv_images.requires_grad = True
        curr_emb = torch.nn.functional.normalize(model(adv_images), dim=1)
        loss = (curr_emb * target_embs).sum()
        model.zero_grad()
        loss.backward()
        adv_images = adv_images - alpha * adv_images.grad.sign()
        eta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + eta, min=-1, max=1).detach()
    return adv_images

print("\n=== Starting CNN Curriculum Training (FGSM -> PGD) ===")
MAX_BATCHES = 2000 

for epoch in range(CONFIG['epochs']):
    student_model.train()
    running_loss = 0.0
    
    current_eps = (epoch / CONFIG['epochs']) * CONFIG['max_epsilon']
    if epoch == 0: current_eps = 0.01 
    
    attack_type = 'FGSM'
    if epoch >= CONFIG['epochs'] // 2: 
        attack_type = 'PGD'
    
    print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Eps: {current_eps:.4f} | Mode: {attack_type}")
    
    loop = tqdm(enumerate(train_loader), total=min(len(train_loader), MAX_BATCHES), leave=True)
    
    for i, (images, _) in loop: 
        if i >= MAX_BATCHES: break
        images = images.to(CONFIG['device'])
        
        with torch.no_grad():
            teacher_emb = torch.nn.functional.normalize(teacher_model(images), dim=1)
        
        if current_eps > 0:
            if attack_type == 'FGSM':
                delta = torch.zeros_like(images, requires_grad=True)
                adv_emb_tmp = torch.nn.functional.normalize(student_model(images + delta), dim=1)
                loss_attack = -(adv_emb_tmp * teacher_emb).sum()
                loss_attack.backward()
                adv_images = images + current_eps * delta.grad.detach().sign()
                adv_images = torch.clamp(adv_images, -1, 1)
                
            elif attack_type == 'PGD':
                adv_images = train_pgd_attack(student_model, images, teacher_emb, current_eps, steps=5)
        else:
            adv_images = images

        optimizer.zero_grad()
        student_adv_emb = torch.nn.functional.normalize(student_model(adv_images), dim=1)
        student_clean_emb = torch.nn.functional.normalize(student_model(images), dim=1)
        loss_robust = 1 - (student_adv_emb * teacher_emb).sum(dim=1).mean()
        loss_clean = 1 - (student_clean_emb * teacher_emb).sum(dim=1).mean()
        loss = 0.5 * loss_robust + 0.5 * loss_clean
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item(), mode=attack_type)

    print(f"Epoch Finished. Avg Loss: {running_loss/(i+1):.4f}")
    torch.save(student_model.state_dict(), "convnext_curriculum_trained.pth")

print("Done! CNN Model saved.")

# %% [code] Cell 7



device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
EPSILONS = [0.0, 0.01, 0.03, 0.05, 0.1]
SQUARE_ITER = 20 

print(f"Running CNN Evaluation on: {device}")

model_name = "hf_hub:gaunernst/convnext_nano.cosface_ms1mv3"
print(f"Loading Model: {model_name}")
model = timm.create_model(model_name, pretrained=False).to(device)

checkpoint_path = "convnext_curriculum_trained.pth"

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    print(f"Weights Loaded from {checkpoint_path}")
else:
    print("WARNING: Model weights file not found! Using random weights.")

model.eval()

valid_gallery_ids = set(gallery_labels_kept)
filtered_probes = []
filtered_labels = []

for face, lbl in zip(probe_faces, probe_labels_kept):
    if lbl in valid_gallery_ids:
        filtered_probes.append(face)
        filtered_labels.append(lbl)

probe_faces_tensor = torch.stack(filtered_probes)
print(f"Valid Test Set: {len(probe_faces_tensor)} images")

print("Computing Gallery Embeddings (CNN)...")
gallery_loader = torch.utils.data.DataLoader(torch.stack(gallery_faces), batch_size=64)
gallery_embs = []
with torch.no_grad():
    for batch in gallery_loader:
        batch = batch.to(device)
        gallery_embs.append(F.normalize(model(batch), dim=1))
gallery_embs = torch.cat(gallery_embs)


def get_metrics(probe_emb, probe_lbls, gallery_emb, gallery_lbls):
    sim_matrix = probe_emb @ gallery_emb.T
    max_scores, max_indices = torch.max(sim_matrix, dim=1)
    correct = 0
    total = len(probe_lbls)
    sim_matrix = sim_matrix.cpu()
    for i in range(total):
        if gallery_lbls[max_indices[i].item()] == probe_lbls[i]:
            correct += 1
    return correct, total

def manual_square_attack(model, images, target_embs, eps, n_queries=20):
    if eps == 0: return images
    adv_images = images.clone().detach()
    
    with torch.no_grad():
        curr_emb = F.normalize(model(adv_images), dim=1)
        best_loss = (curr_emb * target_embs).sum(dim=1) 
    
    for _ in range(n_queries):
        noise = torch.randn_like(images).sign() * eps
        candidate = torch.clamp(adv_images + noise, -1, 1)
        
        with torch.no_grad():
            cand_emb = F.normalize(model(candidate), dim=1)
            cand_loss = (cand_emb * target_embs).sum(dim=1)
            is_better = cand_loss < best_loss
            adv_images[is_better] = candidate[is_better]
            best_loss[is_better] = cand_loss[is_better]
            
    return adv_images


results = []
ATTACKS = ['FGSM', 'Square']

print("\n=== Starting CNN Evaluation ===")

for eps in EPSILONS:
    print(f"\nProcessing Epsilon: {eps}")
    
    num_batches = (len(probe_faces_tensor) + BATCH_SIZE - 1) // BATCH_SIZE
    attack_stats = {atk: {'correct': 0, 'drift': 0.0} for atk in ATTACKS}
    attack_stats['Clean'] = {'correct': 0, 'drift': 0.0}
    
    batch_iterator = tqdm(range(num_batches), desc=f"Eps {eps}", leave=False)
    
    for i in batch_iterator:
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(probe_faces_tensor))
        
        batch_imgs = probe_faces_tensor[start:end].to(device)
        batch_lbls = filtered_labels[start:end]
        
        with torch.no_grad():
            clean_emb = F.normalize(model(batch_imgs), dim=1)
        
        c, t = get_metrics(clean_emb, batch_lbls, gallery_embs, gallery_labels_kept)
        attack_stats['Clean']['correct'] += c
        
        if eps == 0: continue

        for attack_name in ATTACKS:
            if attack_name == 'FGSM':
                img_grad = batch_imgs.clone().detach()
                img_grad.requires_grad = True
                curr = F.normalize(model(img_grad), dim=1)
                loss = (curr * clean_emb).sum() 
                model.zero_grad()
                loss.backward()
                adv_imgs = batch_imgs - eps * img_grad.grad.sign()
                adv_imgs = torch.clamp(adv_imgs, -1, 1)

            elif attack_name == 'Square':
                adv_imgs = manual_square_attack(model, batch_imgs, clean_emb, eps, n_queries=SQUARE_ITER)

            with torch.no_grad():
                adv_emb = F.normalize(model(adv_imgs), dim=1)
            
            acc_c, _ = get_metrics(adv_emb, batch_lbls, gallery_embs, gallery_labels_kept)
            attack_stats[attack_name]['correct'] += acc_c
            
            batch_drift = torch.norm(clean_emb - adv_emb, dim=1).sum().item()
            attack_stats[attack_name]['drift'] += batch_drift

    total_samples = len(probe_faces_tensor)
    clean_acc = attack_stats['Clean']['correct'] / total_samples
    
    if eps == 0:
        for atk in ATTACKS:
             results.append({"Attack": atk, "Epsilon": 0.0, "Accuracy": clean_acc, "ASR": 0.0, "Drift": 0.0})
    else:
        for atk in ATTACKS:
            acc = attack_stats[atk]['correct'] / total_samples
            drift = attack_stats[atk]['drift'] / total_samples
            asr = max(0, clean_acc - acc)
            results.append({"Attack": atk, "Epsilon": eps, "Accuracy": acc, "ASR": asr, "Drift": drift})

df = pd.DataFrame(results)

print("\n" + "="*50)
print("     CNN FINAL RESULTS")
print("="*50)
print(df.round(4))
print("="*50)

df.to_csv("cnn_robustness_results.csv", index=False)
print("Saved to cnn_robustness_results.csv")

fig, axes = plt.subplots(1, 3, figsize=(24, 6))
plt.style.use('seaborn-v0_8-whitegrid')

sns.lineplot(data=df, x="Epsilon", y="Accuracy", hue="Attack", style="Attack", markers=True, ax=axes[0], linewidth=3, markersize=10)
axes[0].set_title("CNN Robustness: Accuracy", fontsize=15, fontweight='bold')
axes[0].set_ylabel("Accuracy")

sns.lineplot(data=df, x="Epsilon", y="ASR", hue="Attack", style="Attack", markers=True, ax=axes[1], linewidth=3, markersize=10)
axes[1].set_title("CNN Vulnerability: ASR", fontsize=15, fontweight='bold')
axes[1].set_ylabel("Accuracy Drop")

sns.lineplot(data=df, x="Epsilon", y="Drift", hue="Attack", style="Attack", markers=True, ax=axes[2], linewidth=3, markersize=10)
axes[2].set_title("CNN Stability: Drift", fontsize=15, fontweight='bold')
axes[2].set_ylabel("L2 Distance")

plt.tight_layout()
plt.savefig('cnn_final_graphs.png')
plt.show()
