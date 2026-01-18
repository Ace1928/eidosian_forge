import functools
import torch
from torch import _C
from torch.onnx import symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('prim::unchecked_cast')
@_beartype.beartype
def prim_unchecked_cast(g: jit_utils.GraphContext, self):
    if isinstance(self.type(), _C.OptionalType):
        return g.op('OptionalGetElement', self)
    return self