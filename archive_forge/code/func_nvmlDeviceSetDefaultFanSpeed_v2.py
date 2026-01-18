from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetDefaultFanSpeed_v2(handle, index):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetDefaultFanSpeed_v2')
    ret = fn(handle, index)
    _nvmlCheckReturn(ret)
    return ret