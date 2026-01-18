from __future__ import annotations
import functools
import sys
import warnings
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.onnx
from torch import _C
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('quantized::group_norm')
@_beartype.beartype
def quantized_group_norm(g: jit_utils.GraphContext, x, num_groups, weight, bias, eps, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    output = opset9.group_norm(g, x, num_groups, weight, bias, eps, False)
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)