from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetPowerManagementLimitConstraints(handle):
    c_minLimit = c_uint()
    c_maxLimit = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerManagementLimitConstraints')
    ret = fn(handle, byref(c_minLimit), byref(c_maxLimit))
    _nvmlCheckReturn(ret)
    return [c_minLimit.value, c_maxLimit.value]