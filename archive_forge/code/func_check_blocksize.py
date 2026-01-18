import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def check_blocksize(f_name, blocksize):
    assert len(blocksize) == 2

    def is_power_of_two(v):
        return not v & v - 1

    def is_compatible_blocksize(b):
        res = True
        for blocksize in b:
            res = (blocksize >= 16 and is_power_of_two(blocksize)) and res
        return res
    check(is_compatible_blocksize(blocksize), f"{f_name}(): sparse inputs' blocksize ({blocksize[0]}, {blocksize[1]}) should be at least 16 and a power of 2 in each dimension.")