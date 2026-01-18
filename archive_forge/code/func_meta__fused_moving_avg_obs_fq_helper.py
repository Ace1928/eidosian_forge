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
@register_meta(aten._fused_moving_avg_obs_fq_helper.default)
def meta__fused_moving_avg_obs_fq_helper(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant=False, symmetric_quant=False):
    torch._check(ch_axis < self.dim(), lambda: 'Error in fused_moving_avg_obs_fake_quant_cpu: ch_axis must be < self.dim()')
    mask = torch.empty_like(self, dtype=torch.bool)
    return (torch.empty_like(self), mask)