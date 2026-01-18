from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetAccountingStats(handle, pid):
    stats = c_nvmlAccountingStats_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAccountingStats')
    ret = fn(handle, c_uint(pid), byref(stats))
    _nvmlCheckReturn(ret)
    if stats.maxMemoryUsage == NVML_VALUE_NOT_AVAILABLE_ulonglong.value:
        stats.maxMemoryUsage = None
    return stats