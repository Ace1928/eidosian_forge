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
def requantize_bias_helper(g: jit_utils.GraphContext, bias, input_scale, weight_scale, axis=None):
    """In PyTorch, bias is float and is quantized to int32 implicitly inside the quantized ATen op kernel.
    In ONNX we need to make the quantization explicit because operators expect all of their inputs to be quantized.
    Since int32 is not a supported output type by ONNX operator `QuantizeLinear`, quantization is exported using
    regular operators.
    """
    bias_scale = g.op('Mul', weight_scale, input_scale)
    bias_scale_shape = g.op('Shape', bias_scale)
    bias_zero_point = g.op('ConstantOfShape', bias_scale_shape, value_t=torch.tensor([0], dtype=torch.int))
    q_bias = g.op('Cast', g.op('Div', bias, bias_scale), to_i=_C_onnx.TensorProtoDataType.INT32)
    axis_args = []
    if axis is not None and (not _is_none(axis)):
        axis_args.append(axis)
    return g.op('prim::TupleConstruct', q_bias, bias_scale, bias_zero_point, *axis_args)