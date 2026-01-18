import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def raggedness_matches(nt, size):
    end = nt._ragged_idx + 1
    return list(nt._size[:end]) == list(size[:end])