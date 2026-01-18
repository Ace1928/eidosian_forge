from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetPowerSource(device):
    c_powerSource = _nvmlPowerSource_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerSource')
    ret = fn(device, byref(c_powerSource))
    _nvmlCheckReturn(ret)
    return c_powerSource.value