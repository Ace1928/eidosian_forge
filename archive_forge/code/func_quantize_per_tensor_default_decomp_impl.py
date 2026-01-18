import functools
import logging
import math
import sys
import typing
from typing import Optional
import torch
import torch._decomp as decomp
import torch._prims_common as utils
import torch.ao.quantization.fx._decomposed
from torch._decomp import (
from torch._decomp.decompositions import (
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import type_to_dtype
from . import config, inductor_prims
@register_decomposition(quantized_decomposed.quantize_per_tensor.default)
def quantize_per_tensor_default_decomp_impl(input: torch.Tensor, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if input.dtype == torch.bfloat16:
        input = input.to(torch.float32)
    inv_scale = 1.0 / scale
    return torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype)