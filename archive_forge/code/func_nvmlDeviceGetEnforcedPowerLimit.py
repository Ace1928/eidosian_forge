from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetEnforcedPowerLimit(handle):
    c_limit = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetEnforcedPowerLimit')
    ret = fn(handle, byref(c_limit))
    _nvmlCheckReturn(ret)
    return c_limit.value