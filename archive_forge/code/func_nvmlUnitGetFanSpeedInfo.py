from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlUnitGetFanSpeedInfo(unit):
    c_speeds = c_nvmlUnitFanSpeeds_t()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetFanSpeedInfo')
    ret = fn(unit, byref(c_speeds))
    _nvmlCheckReturn(ret)
    return c_speeds