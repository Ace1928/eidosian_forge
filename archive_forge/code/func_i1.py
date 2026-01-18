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
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def i1(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i1(a)