from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetVgpuCapabilities(handle, capability):
    c_capResult = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuCapabilities')
    ret = fn(handle, _nvmlDeviceVgpuCapability_t(capability), byref(c_capResult))
    _nvmlCheckReturn(ret)
    return c_capResult.value