from __future__ import annotations
import functools
import sys
from typing import Optional, Tuple
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import (
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::nll_loss2d')
@_beartype.beartype
def nll_loss2d(g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index):
    return nll_loss(g, self, target, weight, reduction, ignore_index)