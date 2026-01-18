from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetTemperatureThreshold(handle, threshold):
    c_temp = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTemperatureThreshold')
    ret = fn(handle, _nvmlTemperatureThresholds_t(threshold), byref(c_temp))
    _nvmlCheckReturn(ret)
    return c_temp.value