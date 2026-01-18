from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetGpuLockedClocks(handle, minGpuClockMHz, maxGpuClockMHz):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetGpuLockedClocks')
    ret = fn(handle, c_uint(minGpuClockMHz), c_uint(maxGpuClockMHz))
    _nvmlCheckReturn(ret)
    return None