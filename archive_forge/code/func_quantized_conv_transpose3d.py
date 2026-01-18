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
@_onnx_symbolic('quantized::conv_transpose3d')
@_beartype.beartype
def quantized_conv_transpose3d(g: jit_utils.GraphContext, q_input, q_weight, bias, stride, padding, output_padding, dilation, groups, op_scale, op_zero_point):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)
    output = opset9.conv_transpose3d(g, input, weight, bias, stride, padding, output_padding, groups, dilation)
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)