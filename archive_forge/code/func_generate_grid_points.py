import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def generate_grid_points():
    for fg, mg in zip(full_grid, grid_blocks):
        yield range(0, fg, mg)