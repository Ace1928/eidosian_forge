from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, minClockMHz, maxClockMHz):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMinMaxClockOfPState')
    ret = fn(device, _nvmlClockType_t(type), _nvmlClockType_t(pstate), minClockMHz, maxClockMHz)
    _nvmlCheckReturn(ret)
    return ret