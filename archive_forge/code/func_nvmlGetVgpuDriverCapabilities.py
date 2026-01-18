from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGetVgpuDriverCapabilities(capability):
    c_capResult = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlGetVgpuDriverCapabilities')
    ret = fn(_nvmlVgpuDriverCapability_t(capability), byref(c_capResult))
    _nvmlCheckReturn(ret)
    return c_capResult.value