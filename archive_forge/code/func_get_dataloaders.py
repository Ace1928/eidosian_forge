import os
from typing import Dict
import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm
import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
def get_dataloaders(batch_size):
    transform = transforms.Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    with FileLock(os.path.expanduser('~/data.lock')):
        training_data = datasets.FashionMNIST(root='~/data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='~/data', train=False, download=True, transform=transform)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return (train_dataloader, test_dataloader)