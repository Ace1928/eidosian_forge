from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetRowRemapperHistogram(device):
    c_vals = c_nvmlRowRemapperHistogramValues()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetRowRemapperHistogram')
    ret = fn(device, byref(c_vals))
    _nvmlCheckReturn(ret)
    return c_vals