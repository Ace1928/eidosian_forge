from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetPciInfo_v3(handle):
    c_info = nvmlPciInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPciInfo_v3')
    ret = fn(handle, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info