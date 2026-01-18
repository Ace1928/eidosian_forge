import argparse
import os
import math
import json
from functools import partial
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import tqdm
import wandb
import numpy as np
from ochat.config import MODEL_CONFIG_MAP
from ochat.training_deepspeed.openchat_dataset import OpenchatDataset
def cosine_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_ratio: float=0.0, num_cycles: float=0.5):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))