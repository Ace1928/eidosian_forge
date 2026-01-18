from __future__ import annotations
from typing import (
import numpy
import onnx
import torch
from torch._subclasses import fake_tensor
def from_sym_value_to_torch_dtype(sym_value: SYM_VALUE_TYPE) -> torch.dtype:
    return _SYM_TYPE_TO_TORCH_DTYPE[type(sym_value)]