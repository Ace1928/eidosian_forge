from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetModuleId(handle, moduleId):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetModuleId')
    ret = fn(handle, moduleId)
    return ret