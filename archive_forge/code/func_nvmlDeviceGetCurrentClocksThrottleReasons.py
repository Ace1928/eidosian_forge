from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetCurrentClocksThrottleReasons(handle):
    c_reasons = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCurrentClocksThrottleReasons')
    ret = fn(handle, byref(c_reasons))
    _nvmlCheckReturn(ret)
    return c_reasons.value