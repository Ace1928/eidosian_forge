from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlUnitGetTemperature(unit, type):
    c_temp = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetTemperature')
    ret = fn(unit, c_uint(type), byref(c_temp))
    _nvmlCheckReturn(ret)
    return c_temp.value