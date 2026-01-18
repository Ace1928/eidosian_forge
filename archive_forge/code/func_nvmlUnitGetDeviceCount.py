from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlUnitGetDeviceCount(unit):
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlUnitGetDevices')
    ret = fn(unit, byref(c_count), None)
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        ret = NVML_SUCCESS
    _nvmlCheckReturn(ret)
    return c_count.value