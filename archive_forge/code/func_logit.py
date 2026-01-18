import math
from typing import Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch import Tensor
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper, out_wrapper
from torch._refs import (
@register_decomposition(aten.logit)
@out_wrapper()
@elementwise_type_promotion_wrapper(type_promoting_args=('self',), type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def logit(self: TensorLikeType, eps: Optional[float]=None) -> TensorLikeType:
    if eps is None:
        eps = -1.0
    lo = eps
    hi = 1 - eps
    self = torch.clamp(self, lo, hi)
    return torch.log(torch.true_divide(self, torch.sub(1, self)))