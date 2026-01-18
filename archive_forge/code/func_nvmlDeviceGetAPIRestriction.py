from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetAPIRestriction(device, apiType):
    c_permission = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAPIRestriction')
    ret = fn(device, _nvmlRestrictedAPI_t(apiType), byref(c_permission))
    _nvmlCheckReturn(ret)
    return int(c_permission.value)