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
@_onnx_symbolic('aten::linalg_vector_norm')
@symbolic_helper.parse_args('v', 'f', 'is', 'b', 'v')
@_beartype.beartype
def linalg_vector_norm(g: jit_utils.GraphContext, self, ord, dim: Optional[Sequence[int]], keepdim: bool, dtype):
    if ord == 0:
        if dim is None:
            self = symbolic_helper._reshape_helper(g, self, g.op('Constant', value_t=torch.tensor([-1], dtype=torch.int64)))
            keepdim = False
        cond_op = g.op('Not', g.op('Equal', self, g.op('Constant', value_t=torch.LongTensor([0]))))
        cond_op = g.op('Cast', cond_op, to_i=_type_utils.JitScalarType.from_value(self).onnx_type())
        return symbolic_helper._reducesum_helper(g, cond_op, axes_i=dim, keepdims_i=keepdim)
    else:
        return opset9.linalg_vector_norm(g, self, ord, dim, keepdim, dtype)