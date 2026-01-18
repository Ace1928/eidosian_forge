import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def local_1d_pattern(attn_size: int, window_size: int) -> torch.Tensor:
    assert window_size % 2 == 1, 'The window size is assumed to be odd (counts self-attention + 2 wings)'
    h_win_size = window_size // 2 + 1
    return local_nd_pattern(attn_size, distance=h_win_size, p=1.0)