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
def is_caffe2_aten_fallback() -> bool:
    return GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK and _C_onnx._CAFFE2_ATEN_FALLBACK