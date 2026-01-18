from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlSystemGetHicVersion():
    c_count = c_uint(0)
    hics = None
    fn = _nvmlGetFunctionPointer('nvmlSystemGetHicVersion')
    ret = fn(byref(c_count), None)
    if ret != NVML_SUCCESS and ret != NVML_ERROR_INSUFFICIENT_SIZE:
        raise NVMLError(ret)
    if c_count.value == 0:
        return []
    hic_array = c_nvmlHwbcEntry_t * c_count.value
    hics = hic_array()
    ret = fn(byref(c_count), hics)
    _nvmlCheckReturn(ret)
    return hics