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
@register_decomposition(aten.rrelu_with_noise_backward)
@out_wrapper()
@pw_cast_for_opmath
def rrelu_with_noise_backward(grad_output: Tensor, self: Tensor, noise: Tensor, lower: float, upper: float, training: bool, self_is_result: bool) -> Tensor:
    if training and upper - lower > 1e-06:
        return grad_output.mul(noise)
    else:
        negative_slope = (lower + upper) / 2
        return aten.leaky_relu_backward(grad_output, self, negative_slope, self_is_result)