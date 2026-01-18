from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, locationType):
    c_count = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemoryErrorCounter')
    ret = fn(handle, _nvmlMemoryErrorType_t(errorType), _nvmlEccCounterType_t(counterType), _nvmlMemoryLocation_t(locationType), byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value