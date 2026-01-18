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
@aten.native_batch_norm.default.py_impl(DispatchKey.Autograd)
@aten.native_batch_norm.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def native_batch_norm_decomposition(input: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool, momentum: float, eps: float) -> Tuple[Tensor, Tensor, Tensor]:
    if running_mean is None and running_var is None:
        return aten._native_batch_norm_legit(input, weight, bias, training, momentum, eps)
    if running_mean is None:
        raise RuntimeError('running_mean is None, but running_var is provided. They should both be None or both be provided.')
    if running_var is None:
        raise RuntimeError('running_var is None, but running_mean is provided. They should both be None or both be provided.')
    if training:
        return aten._native_batch_norm_legit(input, weight, bias, running_mean, running_var, training, momentum, eps)
    else:
        return aten._native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps)