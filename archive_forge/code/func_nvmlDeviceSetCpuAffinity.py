from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetCpuAffinity(handle):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetCpuAffinity')
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None