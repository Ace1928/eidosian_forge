from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance):
    c_frl = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetFrameRateLimit')
    ret = fn(vgpuInstance, byref(c_frl))
    _nvmlCheckReturn(ret)
    return c_frl.value