import functools
import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.cpp_extension import load
@functools.wraps(LoraLinear)
def make_linear_att(*args, **kwargs):
    if 'att' in LORA_CONFIG['parts'] and LORA_CONFIG['r'] > 0:
        return LoraLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)