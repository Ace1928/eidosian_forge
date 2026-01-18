import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def ptr_stride_extractor(*tensors):
    for t in tensors:
        yield t
        yield from t.stride()