from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetAutoBoostedClocksEnabled(handle):
    c_isEnabled = _nvmlEnableState_t()
    c_defaultIsEnabled = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAutoBoostedClocksEnabled')
    ret = fn(handle, byref(c_isEnabled), byref(c_defaultIsEnabled))
    _nvmlCheckReturn(ret)
    return [c_isEnabled.value, c_defaultIsEnabled.value]