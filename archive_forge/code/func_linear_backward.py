import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
@register_meta(aten.linear_backward.default)
def linear_backward(input_, grad_output_, weight_, output_mask):
    grad_input = None
    grad_weight = None
    grad_bias = None
    if output_mask[0]:
        grad_input = grad_output_.new_empty(input_.size())
    if output_mask[1] or output_mask[2]:
        grad_weight = grad_output_.new_empty((grad_output_.size(-1), input_.size(-1)))
        grad_bias = grad_output_.new_empty(grad_output_.size(-1))
    return (grad_input, grad_weight, grad_bias)