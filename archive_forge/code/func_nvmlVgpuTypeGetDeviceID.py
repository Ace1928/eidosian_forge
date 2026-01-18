from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuTypeGetDeviceID(vgpuTypeId):
    c_device_id = c_ulonglong(0)
    c_subsystem_id = c_ulonglong(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetDeviceID')
    ret = fn(vgpuTypeId, byref(c_device_id), byref(c_subsystem_id))
    _nvmlCheckReturn(ret)
    return (c_device_id.value, c_subsystem_id.value)