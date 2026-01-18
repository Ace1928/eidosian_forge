from __future__ import annotations
from typing import (
import numpy
import onnx
import torch
from torch._subclasses import fake_tensor
def from_torch_dtype_to_abbr(dtype: Optional[torch.dtype]) -> str:
    if dtype is None:
        return ''
    return _TORCH_DTYPE_TO_ABBREVIATION.get(dtype, '')