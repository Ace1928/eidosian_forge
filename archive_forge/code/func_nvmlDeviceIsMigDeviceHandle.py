from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceIsMigDeviceHandle(device):
    c_isMigDevice = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceIsMigDeviceHandle')
    ret = fn(device, byref(c_isMigDevice))
    _nvmlCheckReturn(ret)
    return c_isMigDevice