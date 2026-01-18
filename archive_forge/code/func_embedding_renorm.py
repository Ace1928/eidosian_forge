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
@_onnx_symbolic('aten::embedding_renorm')
@symbolic_helper.parse_args('v', 'v', 'f', 'f')
@_beartype.beartype
def embedding_renorm(g: jit_utils.GraphContext, weight, indices, max_norm, norm_type):
    unique_indices = g.op('Unique', indices)
    partial_weight = g.op('Gather', weight, unique_indices)
    norm_i = int(norm_type)
    if norm_i == 1:
        norm_type = 'ReduceL1'
    elif norm_i == 2:
        norm_type = 'ReduceL2'
    else:
        raise errors.SymbolicValueError(f'Unsupported: ONNX export of embedding_renorm with norm: {norm_i}. Only 1. and 2. are supported.', weight)
    partial_weight_norm = g.op(norm_type, partial_weight, axes_i=[1], keepdims_i=1)
    partial_weight_norm_ = g.op('Add', partial_weight_norm, g.op('Constant', value_t=torch.tensor(1e-07)))
    max_norm = torch.tensor(max_norm)
    scales = g.op('Div', max_norm, partial_weight_norm_)
    partial_weight_renorm = g.op('Mul', partial_weight, scales)
    partial_weight_renorm = g.op('Where', g.op('Greater', partial_weight_norm, max_norm), partial_weight_renorm, partial_weight)
    return g.op('ScatterND', weight, symbolic_helper._unsqueeze_helper(g, unique_indices, [1]), partial_weight_renorm)