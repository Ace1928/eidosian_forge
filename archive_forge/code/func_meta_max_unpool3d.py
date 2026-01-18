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
@register_meta(aten.max_unpool3d)
@out_wrapper()
def meta_max_unpool3d(self_, indices, output_size, stride, padding):
    utils.alert_not_deterministic('max_unpooling3d_forward_out')
    _max_unpooling3d_shape_check(self_, indices, output_size, stride, padding, 'max_unpooling3d()')
    self = self_.contiguous()
    odepth, oheight, owidth = output_size
    if self_.ndim == 4:
        nchannels = self.size(0)
        result = self.new_empty((nchannels, odepth, oheight, owidth))
    else:
        nbatch = self.size(0)
        nchannels = self.size(1)
        result = self.new_empty((nbatch, nchannels, odepth, oheight, owidth))
    return result