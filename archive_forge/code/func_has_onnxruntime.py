from torch.onnx._internal.onnxruntime import (
from .registry import register_backend
def has_onnxruntime():
    return is_onnxrt_backend_supported()