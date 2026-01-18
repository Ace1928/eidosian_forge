from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetPowerManagementMode(handle):
    c_pcapMode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerManagementMode')
    ret = fn(handle, byref(c_pcapMode))
    _nvmlCheckReturn(ret)
    return c_pcapMode.value