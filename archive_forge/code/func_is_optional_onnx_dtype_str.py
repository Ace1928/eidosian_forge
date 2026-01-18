from __future__ import annotations
from typing import (
import numpy
import onnx
import torch
from torch._subclasses import fake_tensor
def is_optional_onnx_dtype_str(onnx_type_str: str) -> bool:
    return onnx_type_str in _OPTIONAL_ONNX_DTYPE_STR