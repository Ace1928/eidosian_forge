from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetEccMode(handle, mode):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetEccMode')
    ret = fn(handle, _nvmlEnableState_t(mode))
    _nvmlCheckReturn(ret)
    return None