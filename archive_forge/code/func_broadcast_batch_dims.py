import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def broadcast_batch_dims(f_name, *tensors):
    try:
        return torch.broadcast_shapes(*(t.shape[:-2] for t in tensors))
    except Exception:
        check(False, f"{f_name}(): inputs' batch dimensions are not broadcastable!")