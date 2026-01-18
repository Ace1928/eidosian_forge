from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice):
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDeviceHandleFromMigDeviceHandle')
    ret = fn(migDevice, byref(device))
    _nvmlCheckReturn(ret)
    return device