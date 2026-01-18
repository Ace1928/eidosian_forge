from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetBoardId(handle):
    c_id = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBoardId')
    ret = fn(handle, byref(c_id))
    _nvmlCheckReturn(ret)
    return c_id.value