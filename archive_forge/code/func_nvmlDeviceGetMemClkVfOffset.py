from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMemClkVfOffset(device):
    offset = c_int32()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemClkVfOffset')
    ret = fn(device, byref(offset))
    _nvmlCheckReturn(ret)
    return offset.value