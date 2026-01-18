from __future__ import annotations
import enum
import typing
from typing import Dict, Literal, Optional, Union
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import errors
from torch.onnx._internal import _beartype
@_beartype.beartype
def onnx_compatible(self) -> bool:
    """Return whether this JitScalarType is compatible with ONNX."""
    return self in _SCALAR_TYPE_TO_ONNX and self != JitScalarType.UNDEFINED and (self != JitScalarType.COMPLEX32)