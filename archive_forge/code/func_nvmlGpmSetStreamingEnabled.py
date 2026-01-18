from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpmSetStreamingEnabled(device, state):
    c_state = c_uint(state)
    fn = _nvmlGetFunctionPointer('nvmlGpmSetStreamingEnabled')
    ret = fn(device, c_state)
    _nvmlCheckReturn(ret)
    return ret