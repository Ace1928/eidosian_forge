from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetArchitecture(device):
    arch = _nvmlDeviceArchitecture_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetArchitecture')
    ret = fn(device, byref(arch))
    _nvmlCheckReturn(ret)
    return arch.value