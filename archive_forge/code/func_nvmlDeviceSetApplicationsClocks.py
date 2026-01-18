from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetApplicationsClocks(handle, maxMemClockMHz, maxGraphicsClockMHz):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetApplicationsClocks')
    ret = fn(handle, c_uint(maxMemClockMHz), c_uint(maxGraphicsClockMHz))
    _nvmlCheckReturn(ret)
    return None