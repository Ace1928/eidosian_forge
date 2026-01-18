import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def swin_attention_pattern(H, W, window_size, shift_size=0):
    assert H % window_size == 0
    assert W % window_size == 0
    assert 0 <= shift_size < window_size, 'shift_size must in 0-window_size'
    i, j = _generate_nd_grid(H, W)
    i, j = (i + 0.5, j + 0.5)
    extra = int(shift_size % window_size != 0)
    grid_h = H // window_size + extra
    grid_w = W // window_size + extra
    ii, jj = _generate_nd_grid(grid_h, grid_w)
    s = -shift_size % window_size
    offset = window_size / 2 - s
    ii = ii * window_size + offset
    jj = jj * window_size + offset
    input_coords = torch.stack([i.flatten(), j.flatten()], 1).float()
    anchors_coords = torch.stack([ii.flatten(), jj.flatten()], 1).float()
    anchor_id = torch.cdist(input_coords, anchors_coords, p=2).argmin(1)
    mask = anchor_id[:, None] == anchor_id[None, :]
    return mask