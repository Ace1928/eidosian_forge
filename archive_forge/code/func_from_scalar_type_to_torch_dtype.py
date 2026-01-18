from __future__ import annotations
from typing import (
import numpy
import onnx
import torch
from torch._subclasses import fake_tensor
def from_scalar_type_to_torch_dtype(scalar_type: type) -> Optional[torch.dtype]:
    return _SCALAR_TYPE_TO_TORCH_DTYPE.get(scalar_type)