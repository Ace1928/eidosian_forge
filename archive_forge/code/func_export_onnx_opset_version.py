import torch._C._onnx as _C_onnx
from torch.onnx import _constants
@export_onnx_opset_version.setter
def export_onnx_opset_version(self, value: int):
    supported_versions = range(_constants.ONNX_MIN_OPSET, _constants.ONNX_MAX_OPSET + 1)
    if value not in supported_versions:
        raise ValueError(f'Unsupported ONNX opset version: {value}')
    self._export_onnx_opset_version = value