from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetCudaComputeCapability(handle):
    c_major = c_int()
    c_minor = c_int()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCudaComputeCapability')
    ret = fn(handle, byref(c_major), byref(c_minor))
    _nvmlCheckReturn(ret)
    return (c_major.value, c_minor.value)