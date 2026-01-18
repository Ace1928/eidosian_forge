from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetComputeInstanceId(device):
    c_computeInstanceId = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetComputeInstanceId')
    ret = fn(device, byref(c_computeInstanceId))
    _nvmlCheckReturn(ret)
    return c_computeInstanceId.value