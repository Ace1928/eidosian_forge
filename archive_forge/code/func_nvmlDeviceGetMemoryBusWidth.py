from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMemoryBusWidth(device):
    c_memBusWidth = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemoryBusWidth')
    ret = fn(device, byref(c_memBusWidth))
    _nvmlCheckReturn(ret)
    return c_memBusWidth.value