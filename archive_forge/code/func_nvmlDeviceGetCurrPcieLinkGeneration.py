from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetCurrPcieLinkGeneration(handle):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCurrPcieLinkGeneration')
    gen = c_uint()
    ret = fn(handle, byref(gen))
    _nvmlCheckReturn(ret)
    return gen.value