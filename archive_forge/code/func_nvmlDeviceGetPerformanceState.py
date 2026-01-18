from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetPerformanceState(handle):
    c_pstate = _nvmlPstates_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPerformanceState')
    ret = fn(handle, byref(c_pstate))
    _nvmlCheckReturn(ret)
    return c_pstate.value