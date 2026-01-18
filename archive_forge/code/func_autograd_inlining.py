import torch._C._onnx as _C_onnx
from torch.onnx import _constants
@autograd_inlining.setter
def autograd_inlining(self, value: bool):
    if type(value) is not bool:
        raise TypeError('autograd_inlining must be a boolean')
    self._autograd_inlining = value