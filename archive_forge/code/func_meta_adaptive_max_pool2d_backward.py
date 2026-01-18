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
@register_meta(aten.adaptive_max_pool2d_backward)
@out_wrapper('grad_input')
def meta_adaptive_max_pool2d_backward(grad_output, input, indices):
    ndim = grad_output.ndim
    torch._check(ndim in (3, 4), lambda: f'adaptive_max_pooling2d_backward(): Expected 3D or 4D grad_output, but got: {grad_output.shape}')
    _adaptive_pool_empty_output_check(grad_output, 'adaptive_max_pool2d_backward')
    torch._check(input.dtype == grad_output.dtype, lambda: f'expected dtype {input.dtype} for `grad_output` but got dtype {grad_output.dtype}')
    memory_format = utils.suggest_memory_format(input)
    return input.new_empty(input.shape).to(memory_format=memory_format)