from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId):
    c_frl_config = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetFrameRateLimit')
    ret = fn(vgpuTypeId, byref(c_frl_config))
    _nvmlCheckReturn(ret)
    return c_frl_config.value