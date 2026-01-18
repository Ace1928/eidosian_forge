from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetClock(handle, type, id):
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetClock')
    ret = fn(handle, _nvmlClockType_t(type), _nvmlClockId_t(id), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value