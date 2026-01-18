from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetNumGpuCores(device):
    c_numCores = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNumGpuCores')
    ret = fn(device, byref(c_numCores))
    _nvmlCheckReturn(ret)
    return c_numCores.value