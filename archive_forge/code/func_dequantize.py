from __future__ import annotations
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
@_tensor_operation
def dequantize(self, input, scale, shift, nbit, dst_ty=None):
    if dst_ty is None:
        dst_ty = torch.float16
    raise NotImplementedError()