import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func([torch.ops.aten.size.default], 'self: jt_all')
def tensor_attr_unsupported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.size.default:
        raise RuntimeError('NestedTensors does not support directly calling torch.ops.aten.size please use `nested_tensor.size()` instead.')