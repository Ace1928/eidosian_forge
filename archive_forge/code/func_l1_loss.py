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
@elementwise_type_promotion_wrapper(type_promoting_args=('input', 'target'), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT)
def l1_loss(input: TensorLikeType, target: TensorLikeType, size_average: Optional[bool]=None, reduce: Optional[bool]=None, reduction: str='mean') -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.l1_loss
    """
    if size_average is not None or reduce is not None:
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    _check_reduction_value(reduction)
    loss = torch.abs(input - target)
    return _apply_loss_reduction(loss, reduction)