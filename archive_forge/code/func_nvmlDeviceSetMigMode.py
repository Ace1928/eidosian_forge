from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetMigMode(device, mode):
    c_activationStatus = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetMigMode')
    ret = fn(device, mode, byref(c_activationStatus))
    _nvmlCheckReturn(ret)
    return c_activationStatus.value