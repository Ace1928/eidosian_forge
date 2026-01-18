import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def check_ragged_dim_same(func, a: NestedTensor, a_name: str, b: NestedTensor, b_name: str) -> None:
    if a._size[a._ragged_idx] != b._size[b._ragged_idx]:
        raise RuntimeError(f'NestedTensor {func.__name__}: expected {a_name} and {b_name} to have the same exact offsets tensor.')