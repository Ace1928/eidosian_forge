from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlUnitGetCount():
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetCount')
    ret = fn(byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value