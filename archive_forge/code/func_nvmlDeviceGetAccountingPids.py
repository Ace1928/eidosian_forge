from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetAccountingPids(handle):
    count = c_uint(nvmlDeviceGetAccountingBufferSize(handle))
    pids = (c_uint * count.value)()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAccountingPids')
    ret = fn(handle, byref(count), pids)
    _nvmlCheckReturn(ret)
    return list(map(int, pids[0:count.value]))