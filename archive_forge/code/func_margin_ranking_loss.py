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
@register_decomposition(aten.margin_ranking_loss)
def margin_ranking_loss(input1: TensorLikeType, input2: TensorLikeType, target: TensorLikeType, margin: float=0.0, reduction: str='mean') -> TensorLikeType:
    if input1.ndim != input2.ndim or input1.ndim != target.ndim:
        raise RuntimeError(f'margin_ranking_loss : All input tensors should have same dimension but got sizes: input1: {input1.shape}, input2: {input2.shape}, target: {target.shape} ')
    _check_reduction_value(reduction)
    loss = torch.clamp_min(-target * (input1 - input2) + margin, 0)
    return _apply_loss_reduction(loss, reduction)