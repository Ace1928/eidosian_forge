import functools
import torch
from torch import _C
from torch.onnx import symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::__is_')
@_beartype.beartype
def aten__is_(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_none(other):
        if isinstance(self.type(), _C.OptionalType):
            none = g.op('OptionalHasElement', self)
            return g.op('Not', none)
        else:
            return g.op('Constant', value_t=torch.BoolTensor([0]))
    return opset9.eq(g, self, other)