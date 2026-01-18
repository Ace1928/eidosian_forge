from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMigDeviceHandleByIndex(device, index):
    c_index = c_uint(index)
    migDevice = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMigDeviceHandleByIndex')
    ret = fn(device, c_index, byref(migDevice))
    _nvmlCheckReturn(ret)
    return migDevice