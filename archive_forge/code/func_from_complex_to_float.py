from __future__ import annotations
from typing import (
import numpy
import onnx
import torch
from torch._subclasses import fake_tensor
def from_complex_to_float(dtype: torch.dtype) -> torch.dtype:
    return _COMPLEX_TO_FLOAT[dtype]