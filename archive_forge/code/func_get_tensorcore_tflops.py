import heapq
import torch
from .. import cdiv
from .._C.libtriton.triton import runtime
from ..runtime import driver
from ..testing import (get_dram_gbps, get_max_simd_tflops, get_max_tensorcore_tflops, nvsmi)
def get_tensorcore_tflops(backend, device, num_ctas, num_warps, dtype):
    """ return compute throughput in TOPS """
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = driver.utils.get_device_properties(device)['multiprocessor_count'] * 4
    cur_sm_clock = nvsmi(['clocks.max.sm'])[0]
    tflops = min(num_subcores, total_warps) / num_subcores * get_max_tensorcore_tflops(dtype, cur_sm_clock, backend, device)
    return tflops