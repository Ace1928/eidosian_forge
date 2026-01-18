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
def op_wrapper(input, axis_i, keepdims_i):
    if g.opset >= 12:
        return g.op(op_name, input, axis_i=axis_i, keepdims_i=keepdims_i, select_last_index_i=False)
    return g.op(op_name, input, axis_i=axis_i, keepdims_i=keepdims_i)