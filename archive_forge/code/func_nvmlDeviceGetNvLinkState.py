from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetNvLinkState(device, link):
    c_isActive = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkState')
    ret = fn(device, link, byref(c_isActive))
    _nvmlCheckReturn(ret)
    return c_isActive.value