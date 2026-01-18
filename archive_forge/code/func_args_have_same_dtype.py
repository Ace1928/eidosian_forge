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
def args_have_same_dtype(args):
    assert args
    base_dtype = _type_utils.JitScalarType.from_value(args[0])
    has_same_dtype = all((_type_utils.JitScalarType.from_value(elem) == base_dtype for elem in args))
    return has_same_dtype