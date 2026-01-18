from __future__ import annotations
import logging
from typing import Optional, Sequence
import torch
from torch import _prims, Tensor
def eager_force_stride(input_tensor: Tensor, stride) -> Tensor:
    if input_tensor.stride() == stride:
        return input_tensor
    new_tensor = input_tensor.clone().as_strided(input_tensor.shape, stride)
    new_tensor.copy_(input_tensor)
    return new_tensor