from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetAutoBoostedClocksEnabled(handle, enabled):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetAutoBoostedClocksEnabled')
    ret = fn(handle, _nvmlEnableState_t(enabled))
    _nvmlCheckReturn(ret)
    return None