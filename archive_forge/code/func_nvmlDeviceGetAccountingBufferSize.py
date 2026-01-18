from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetAccountingBufferSize(handle):
    bufferSize = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAccountingBufferSize')
    ret = fn(handle, byref(bufferSize))
    _nvmlCheckReturn(ret)
    return int(bufferSize.value)