from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetPowerUsage(handle):
    c_watts = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerUsage')
    ret = fn(handle, byref(c_watts))
    _nvmlCheckReturn(ret)
    return c_watts.value