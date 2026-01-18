from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetComputeMode(handle):
    c_mode = _nvmlComputeMode_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetComputeMode')
    ret = fn(handle, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value