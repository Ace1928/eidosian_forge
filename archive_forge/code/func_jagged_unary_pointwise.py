import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def jagged_unary_pointwise(func, *args, **kwargs):
    return NestedTensor(func(args[0]._values, *args[1:], **kwargs), **extract_kwargs(args[0]))