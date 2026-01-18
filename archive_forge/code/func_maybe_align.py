import dropout_layer_norm
import torch
from torch.nn import init
def maybe_align(x, alignment_in_bytes=16):
    """Assume that x already has last dim divisible by alignment_in_bytes"""
    return x if x.data_ptr() % alignment_in_bytes == 0 else x.clone()