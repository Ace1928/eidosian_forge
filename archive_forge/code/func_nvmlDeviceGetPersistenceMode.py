from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetPersistenceMode(handle):
    c_state = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPersistenceMode')
    ret = fn(handle, byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state.value