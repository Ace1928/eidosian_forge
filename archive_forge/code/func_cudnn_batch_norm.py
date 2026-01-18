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
@aten.cudnn_batch_norm.default.py_impl(DispatchKey.Autograd)
@register_decomposition(aten.cudnn_batch_norm)
@out_wrapper('out0', 'out1', 'out2', 'out3')
def cudnn_batch_norm(input: Tensor, weight: Tensor, bias: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool, exponential_average_factor: float, epsilon: float):
    a, b, c = aten.native_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
    if training:
        return (a, b, c, input.new_zeros((0,), dtype=torch.uint8))
    return (a, weight.new_zeros((0,)), weight.new_zeros((0,)), input.new_zeros((0,), dtype=torch.uint8))