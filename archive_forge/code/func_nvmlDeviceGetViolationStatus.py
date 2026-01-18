from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetViolationStatus(device, perfPolicyType):
    c_perfPolicy_type = _nvmlPerfPolicyType_t(perfPolicyType)
    c_violTime = c_nvmlViolationTime_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetViolationStatus')
    ret = fn(device, c_perfPolicy_type, byref(c_violTime))
    _nvmlCheckReturn(ret)
    return c_violTime