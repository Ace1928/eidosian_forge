from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetAdaptiveClockInfoStatus(device):
    c_adaptiveClockInfoStatus = _nvmlAdaptiveClockInfoStatus_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAdaptiveClockInfoStatus')
    ret = fn(device, byref(c_adaptiveClockInfoStatus))
    _nvmlCheckReturn(ret)
    return c_adaptiveClockInfoStatus.value