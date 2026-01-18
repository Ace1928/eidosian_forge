import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.is_same_size.default, 'self: jt, other: jt')
def is_same_size_default(func, *args, **kwargs):
    return args[0]._size == args[1]._size