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
def should_fold(tensor1: torch.Tensor, tensor2: torch.Tensor) -> bool:
    t1, t2 = (tensor1, tensor2) if tensor1.ndim >= tensor2.ndim else (tensor2, tensor1)
    if not (t1.ndim >= 3 and t2.ndim <= 2):
        return False
    if t2.requires_grad:
        return True
    if tensor1.ndim == 2:
        return False
    if t1.numel() == 0:
        return True
    t1_shape = t1.shape
    t1_stride = t1.stride()
    return all((st1 == st2 * s2 for st1, st2, s2 in zip(t1_stride[:-2], t1_stride[1:-1], t1_shape[1:-1])))