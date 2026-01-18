import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def get_gpu_memory(gpu_dev_id):
    free_mem = ctypes.c_uint64(0)
    total_mem = ctypes.c_uint64(0)
    check_call(_LIB.MXGetGPUMemoryInformation64(gpu_dev_id, ctypes.byref(free_mem), ctypes.byref(total_mem)))
    return (free_mem.value, total_mem.value)