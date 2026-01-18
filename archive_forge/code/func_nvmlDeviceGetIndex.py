from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetIndex(handle):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetIndex')
    c_index = c_uint()
    ret = fn(handle, byref(c_index))
    _nvmlCheckReturn(ret)
    return c_index.value