import argparse
from enum import Enum
import importlib
import logging
import tempfile
import time
from typing import Any, List, Optional, cast
from golden_configs import oss_mnist
import numpy as np
import torch
import torch.autograd.profiler as profiler
from torch.cuda.amp import GradScaler as TorchGradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from benchmarks.datasets.mnist import setup_cached_mnist
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
def run_closure(closure, scaler, optimizer):
    if scaler is not None:
        final_loss = closure(grad_scaler=scaler)
        scaler.step(optimizer)
        scaler.update()
        return final_loss
    else:
        return optimizer.step(closure)