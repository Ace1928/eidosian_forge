from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetVirtualizationMode(handle):
    c_virtualization_mode = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVirtualizationMode')
    ret = fn(handle, byref(c_virtualization_mode))
    _nvmlCheckReturn(ret)
    return c_virtualization_mode.value