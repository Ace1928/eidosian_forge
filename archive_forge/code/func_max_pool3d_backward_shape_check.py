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
def max_pool3d_backward_shape_check(input, grad_output, indices, nslices, kT, kH, kW, dT, dH, dW, pT, pH, pW, dilationT, dilationH, dilationW, itime, iheight, iwidth, otime, oheight, owidth, fn_name):
    ndim = input.ndim
    pool3d_shape_check(input, nslices, kT, kH, kW, dT, dH, dW, pT, pH, pW, dilationT, dilationH, dilationW, itime, iheight, iwidth, otime, oheight, owidth, fn_name)
    check_dim_size(grad_output, ndim, ndim - 4, nslices)
    check_dim_size(grad_output, ndim, ndim - 3, otime)
    check_dim_size(grad_output, ndim, ndim - 2, oheight)
    check_dim_size(grad_output, ndim, ndim - 1, owidth)
    check_dim_size(indices, ndim, ndim - 4, nslices)
    check_dim_size(indices, ndim, ndim - 3, otime)
    check_dim_size(indices, ndim, ndim - 2, oheight)
    check_dim_size(indices, ndim, ndim - 1, owidth)