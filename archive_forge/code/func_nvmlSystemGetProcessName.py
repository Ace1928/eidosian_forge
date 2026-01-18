from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlSystemGetProcessName(pid):
    c_name = create_string_buffer(1024)
    fn = _nvmlGetFunctionPointer('nvmlSystemGetProcessName')
    ret = fn(c_uint(pid), c_name, c_uint(1024))
    _nvmlCheckReturn(ret)
    return c_name.value