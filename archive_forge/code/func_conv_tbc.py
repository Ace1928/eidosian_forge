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
@_onnx_symbolic('aten::conv_tbc')
@symbolic_helper.parse_args('v', 'v', 'v', 'i')
@_beartype.beartype
def conv_tbc(g: jit_utils.GraphContext, input, weight, bias, pad):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at('conv_tbc', input, weight, bias, pad_i=pad)
    else:
        input = g.op('Transpose', input, perm_i=[1, 2, 0])
        weight = g.op('Transpose', weight, perm_i=[2, 1, 0])
        conv = conv1d(g, input, weight, bias, [1], [pad], [1], 1)
        return g.op('Transpose', conv, perm_i=[2, 0, 1])