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
@_onnx_symbolic('aten::replication_pad1d')
@_onnx_symbolic('aten::replication_pad2d')
@_onnx_symbolic('aten::replication_pad3d')
@_beartype.beartype
def replication_pad(g: jit_utils.GraphContext, input, padding):
    mode = 'edge'
    paddings = _prepare_onnx_paddings(g, input, padding)
    return g.op('Pad', input, paddings, mode_s=mode)