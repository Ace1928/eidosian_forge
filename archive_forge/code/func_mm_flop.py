import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
@register_flop_formula(aten.mm)
def mm_flop(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for matmul."""
    m, k = a_shape
    k2, n = b_shape
    assert k == k2
    return m * n * 2 * k