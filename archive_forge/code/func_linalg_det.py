from __future__ import annotations
import functools
import sys
import warnings
from typing import Optional, Sequence
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::linalg_det')
@_beartype.beartype
def linalg_det(g: jit_utils.GraphContext, self):
    return g.op('Det', self)