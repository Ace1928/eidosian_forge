from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetPcieLinkMaxSpeed(device):
    c_speed = _nvmlPcieLinkMaxSpeed_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPcieLinkMaxSpeed')
    ret = fn(device, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value