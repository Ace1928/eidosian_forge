import functools
import torch
from torch import _C
from torch.onnx import symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::__isnot_')
@opset9.wrap_logical_op_with_negation
@_beartype.beartype
def aten__isnot_(g: jit_utils.GraphContext, self, other):
    return aten__is_(g, self, other)