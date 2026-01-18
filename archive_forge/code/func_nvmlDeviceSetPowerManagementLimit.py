from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetPowerManagementLimit(handle, limit):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetPowerManagementLimit')
    ret = fn(handle, c_uint(limit))
    _nvmlCheckReturn(ret)
    return None