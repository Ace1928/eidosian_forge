from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetSupportedMemoryClocks(handle):
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSupportedMemoryClocks')
    ret = fn(handle, byref(c_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        clocks_array = c_uint * c_count.value
        c_clocks = clocks_array()
        ret = fn(handle, byref(c_count), c_clocks)
        _nvmlCheckReturn(ret)
        procs = []
        for i in range(c_count.value):
            procs.append(c_clocks[i])
        return procs
    else:
        raise NVMLError(ret)