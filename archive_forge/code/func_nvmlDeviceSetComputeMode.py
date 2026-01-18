from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetComputeMode(handle, mode):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetComputeMode')
    ret = fn(handle, _nvmlComputeMode_t(mode))
    _nvmlCheckReturn(ret)
    return None