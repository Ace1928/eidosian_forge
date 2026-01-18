import math
import os
import torch
from filelock import FileLock
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray import train, tune
def prepare_data(self):
    MNIST(self.data_dir, train=True, download=True)
    MNIST(self.data_dir, train=False, download=True)