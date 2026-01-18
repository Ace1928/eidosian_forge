from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlEventSetWait_v2(eventSet, timeoutms):
    fn = _nvmlGetFunctionPointer('nvmlEventSetWait_v2')
    data = c_nvmlEventData_t()
    ret = fn(eventSet, byref(data), c_uint(timeoutms))
    _nvmlCheckReturn(ret)
    return data