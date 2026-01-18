from __future__ import annotations
from typing import (
import numpy
import onnx
import torch
from torch._subclasses import fake_tensor
def is_torch_symbolic_type(value: Any) -> bool:
    return isinstance(value, (torch.SymBool, torch.SymInt, torch.SymFloat))