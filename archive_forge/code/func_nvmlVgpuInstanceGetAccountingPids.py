from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetAccountingPids(vgpuInstance):
    c_pidCount = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetAccountingPids')
    ret = fn(vgpuInstance, byref(c_pidCount), None)
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        sampleArray = c_pidCount.value * c_uint
        c_pidArray = sampleArray()
        ret = fn(vgpuInstance, byref(c_pidCount), byref(c_pidArray))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return (c_pidCount, c_pidArray)