from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetBusType(device):
    c_busType = _nvmlBusType_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBusType')
    ret = fn(device, byref(c_busType))
    _nvmlCheckReturn(ret)
    return c_busType.value