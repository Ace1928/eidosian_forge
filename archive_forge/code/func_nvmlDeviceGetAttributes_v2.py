from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetAttributes_v2(device):
    c_attrs = c_nvmlDeviceAttributes()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAttributes_v2')
    ret = fn(device, byref(c_attrs))
    _nvmlCheckReturn(ret)
    return c_attrs