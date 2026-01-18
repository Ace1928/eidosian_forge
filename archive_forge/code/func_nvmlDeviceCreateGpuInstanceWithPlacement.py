from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, placement):
    c_instance = c_nvmlGpuInstance_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceCreateGpuInstanceWithPlacement')
    ret = fn(device, profileId, placement, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance