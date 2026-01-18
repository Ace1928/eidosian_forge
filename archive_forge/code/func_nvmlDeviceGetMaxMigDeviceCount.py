from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMaxMigDeviceCount(device):
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMaxMigDeviceCount')
    ret = fn(device, byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value