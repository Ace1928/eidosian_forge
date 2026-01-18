from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMaxPcieLinkWidth(handle):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMaxPcieLinkWidth')
    width = c_uint()
    ret = fn(handle, byref(width))
    _nvmlCheckReturn(ret)
    return width.value