from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGspFirmwareMode(handle, isEnabled, defaultMode):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGspFirmwareMode')
    ret = fn(handle, isEnabled, defaultMode)
    _nvmlCheckReturn(ret)
    return ret