from __future__ import annotations
import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number
@_onnx_symbolic('aten::linalg_matrix_norm')
@symbolic_helper.parse_args('v', 'v', 'is', 'b', 'v')
@_beartype.beartype
def linalg_matrix_norm(g: jit_utils.GraphContext, self: torch._C.Value, ord: torch._C.Value, dim: List[int], keepdim: bool, dtype: torch._C.Value):
    ord_value = symbolic_helper._parse_arg(ord, 's')
    if ord_value == 'fro':
        return frobenius_norm(g, self, dim, keepdim)
    elif ord_value == 'nuc':
        return symbolic_helper._unimplemented('linalg.matrix_norm', 'ord==nuc', self)
    else:
        ord_value = symbolic_helper._parse_arg(ord, 'f')
        if ord_value is None:
            return frobenius_norm(g, self, dim, keepdim)
        if ord_value == 2 or ord_value == -2:
            return symbolic_helper._unimplemented('linalg.matrix_norm', 'ord==2', self)
        self_dim = symbolic_helper._get_tensor_rank(self)
        if self_dim is None:
            return symbolic_helper._unimplemented('linalg.matrix_norm', 'Input rank must be known at export time.', self)
        if dim[0] < 0:
            dim[0] += self_dim
        if dim[1] < 0:
            dim[1] += self_dim
        if ord_value == math.inf or ord_value == -math.inf:
            dim[0], dim[1] = (dim[1], dim[0])
        if dim[1] > dim[0] and (not keepdim):
            dim[1] -= 1
        sum = symbolic_helper._reducesum_helper(g, g.op('Abs', self), axes_i=[dim[0]], keepdims_i=keepdim)
        if ord_value > 0:
            result, indices = max(g, sum, dim_or_y=g.op('Constant', value_t=torch.LongTensor([dim[1]])), keepdim=keepdim)
        else:
            result, indices = min(g, sum, dim_or_y=g.op('Constant', value_t=torch.LongTensor([dim[1]])), keepdim=keepdim)
        return result