import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
@staticmethod
def attn_mask_postprocess(attention_mask):
    attention_mask_b1s = attention_mask.unsqueeze(1)
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    extended_attention_mask = attention_mask_bss < 0.5
    return extended_attention_mask