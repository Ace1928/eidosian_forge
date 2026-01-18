from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetPcieThroughput(device, counter):
    c_util = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPcieThroughput')
    ret = fn(device, _nvmlPcieUtilCounter_t(counter), byref(c_util))
    _nvmlCheckReturn(ret)
    return c_util.value