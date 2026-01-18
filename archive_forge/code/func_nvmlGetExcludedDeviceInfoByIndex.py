from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGetExcludedDeviceInfoByIndex(index):
    c_index = c_uint(index)
    info = c_nvmlExcludedDeviceInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlGetExcludedDeviceInfoByIndex')
    ret = fn(c_index, byref(info))
    _nvmlCheckReturn(ret)
    return info