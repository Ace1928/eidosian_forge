from __future__ import annotations
import functools
import inspect
import sys
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _type_utils, errors
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils
from torch.types import Number
@_beartype.beartype
def is_complex_value(x: _C.Value) -> bool:
    assert _is_value(x)
    return _type_utils.JitScalarType.from_value(x, _type_utils.JitScalarType.UNDEFINED) in {_type_utils.JitScalarType.COMPLEX32, _type_utils.JitScalarType.COMPLEX64, _type_utils.JitScalarType.COMPLEX128}