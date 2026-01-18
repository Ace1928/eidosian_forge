from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetFBCStats(handle):
    c_fbcStats = c_nvmlFBCStats_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetFBCStats')
    ret = fn(handle, byref(c_fbcStats))
    _nvmlCheckReturn(ret)
    return c_fbcStats