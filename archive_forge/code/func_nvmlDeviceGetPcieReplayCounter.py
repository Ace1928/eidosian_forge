from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetPcieReplayCounter(handle):
    c_replay = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPcieReplayCounter')
    ret = fn(handle, byref(c_replay))
    _nvmlCheckReturn(ret)
    return c_replay.value