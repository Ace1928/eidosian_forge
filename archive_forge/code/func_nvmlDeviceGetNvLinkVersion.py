from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetNvLinkVersion(device, link):
    c_version = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkVersion')
    ret = fn(device, link, byref(c_version))
    _nvmlCheckReturn(ret)
    return c_version.value