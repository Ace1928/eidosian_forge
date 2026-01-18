import math
from functools import wraps
from typing import Callable, Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _make_inplace
@register_decomposition(aten.hinge_embedding_loss)
def hinge_embedding_loss(input: TensorLikeType, target: TensorLikeType, margin: float=1.0, reduction: str='mean') -> TensorLikeType:
    _check_reduction_value(reduction)
    margin_clamp = torch.clamp_min(margin - input, 0)
    output_margin = torch.where(target != 1, margin_clamp, 0)
    output_self = torch.where(target != -1, input, 0)
    loss = output_margin + output_self
    return _apply_loss_reduction(loss, reduction)