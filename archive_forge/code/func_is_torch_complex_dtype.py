from __future__ import annotations
from typing import (
import numpy
import onnx
import torch
from torch._subclasses import fake_tensor
def is_torch_complex_dtype(tensor_dtype: torch.dtype) -> bool:
    return tensor_dtype in _COMPLEX_TO_FLOAT