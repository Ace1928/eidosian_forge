import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def dilated_2d_pattern(H, W, k=2):
    """
    Returns a 2d pattern that samples 1 every k elements in the attention mask.
    Can be seen as a form of downsampling, where every pixel attends to a downsampled
    version of the input.
    """
    d_h = local_nd_distance(H, W, p=1, weights=(1, 0))
    d_w = local_nd_distance(H, W, p=1, weights=(0, 1))
    d = (d_h.floor() % k == 0) & (d_w.floor() % k == 0)
    return d