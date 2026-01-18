from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetConfComputeMemSizeInfo(device):
    c_ccMemSize = c_nvmlConfComputeMemSizeInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetConfComputeMemSizeInfo')
    ret = fn(device, byref(c_ccMemSize))
    _nvmlCheckReturn(ret)
    return c_ccMemSize