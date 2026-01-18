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
@_onnx_symbolic('aten::conv_transpose1d')
@symbolic_helper.parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'is')
@_beartype.beartype
def conv_transpose1d(g: jit_utils.GraphContext, input, weight, bias, stride, padding, output_padding, groups, dilation):
    return _convolution(g, input, weight, bias, stride, padding, dilation, True, output_padding, groups, None, None, None, None)