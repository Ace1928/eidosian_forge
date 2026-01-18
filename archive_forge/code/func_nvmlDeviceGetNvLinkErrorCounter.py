from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetNvLinkErrorCounter(device, link, counter):
    c_result = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkErrorCounter')
    ret = fn(device, link, counter, byref(c_result))
    _nvmlCheckReturn(ret)
    return c_result.value