import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
def compute_idx(in_size, out_size):
    orange = torch.arange(out_size, device=device, dtype=torch.int64)
    i0 = start_index(orange, out_size, in_size)
    maxlength = in_size // out_size + 1
    in_size_mod = in_size % out_size
    adaptive = not (in_size_mod == 0 or out_size % in_size_mod == 0)
    if adaptive:
        maxlength += 1
    elif in_size_mod == 0:
        maxlength -= 1
    range_max = torch.arange(maxlength, device=device, dtype=torch.int64)
    idx = i0.unsqueeze(-1) + range_max
    if adaptive:
        maxval = torch.scalar_tensor(in_size - 1, dtype=idx.dtype, device=idx.device)
        idx = torch.minimum(idx, maxval)
        i1 = end_index(orange, out_size, in_size)
        length = i1 - i0
    else:
        length = maxlength
    return (idx, length, range_max, adaptive)