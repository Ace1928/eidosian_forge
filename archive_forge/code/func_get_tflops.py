import heapq
import torch
from .. import cdiv
from .._C.libtriton.triton import runtime
from ..runtime import driver
from ..testing import (get_dram_gbps, get_max_simd_tflops, get_max_tensorcore_tflops, nvsmi)
def get_tflops(backend, device, num_ctas, num_warps, dtype):
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8 and dtype == torch.float32:
        return get_simd_tflops(backend, device, num_ctas, num_warps, dtype)
    return get_tensorcore_tflops(backend, device, num_ctas, num_warps, dtype)