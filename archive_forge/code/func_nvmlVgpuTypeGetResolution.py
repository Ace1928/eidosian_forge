from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuTypeGetResolution(vgpuTypeId):
    c_xdim = c_uint(0)
    c_ydim = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetResolution')
    ret = fn(vgpuTypeId, 0, byref(c_xdim), byref(c_ydim))
    _nvmlCheckReturn(ret)
    return (c_xdim.value, c_ydim.value)