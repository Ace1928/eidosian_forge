import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def is_compatible_blocksize(b):
    res = True
    for blocksize in b:
        res = (blocksize >= 16 and is_power_of_two(blocksize)) and res
    return res