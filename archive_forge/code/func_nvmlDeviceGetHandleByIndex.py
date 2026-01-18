from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetHandleByIndex(index):
    c_index = c_uint(index)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetHandleByIndex_v2')
    ret = fn(c_index, byref(device))
    _nvmlCheckReturn(ret)
    return device