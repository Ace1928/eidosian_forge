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
def state_dict_to_cpu(item, device=torch.device('cpu')):
    if torch.is_tensor(item):
        return item.detach().to(device)
    elif isinstance(item, list):
        return [state_dict_to_cpu(v, device) for v in item]
    elif isinstance(item, tuple):
        return tuple([state_dict_to_cpu(v, device) for v in item])
    elif isinstance(item, dict):
        return type(item)({k: state_dict_to_cpu(v, device) for k, v in item.items()})
    else:
        return item