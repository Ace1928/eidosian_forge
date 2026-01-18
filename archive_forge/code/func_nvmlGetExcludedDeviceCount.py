from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGetExcludedDeviceCount():
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlGetExcludedDeviceCount')
    ret = fn(byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value