import torch._C._onnx as _C_onnx
from torch.onnx import _constants
@in_onnx_export.setter
def in_onnx_export(self, value: bool):
    if type(value) is not bool:
        raise TypeError('in_onnx_export must be a boolean')
    self._in_onnx_export = value