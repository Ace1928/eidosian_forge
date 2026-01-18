from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetProcessUtilization(handle, timeStamp):
    c_count = c_uint(0)
    c_time_stamp = c_ulonglong(timeStamp)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetProcessUtilization')
    ret = fn(handle, None, byref(c_count), c_time_stamp)
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        sampleArray = c_count.value * c_nvmlProcessUtilizationSample_t
        c_samples = sampleArray()
        ret = fn(handle, c_samples, byref(c_count), c_time_stamp)
        _nvmlCheckReturn(ret)
        return c_samples[0:c_count.value]
    else:
        raise NVMLError(ret)