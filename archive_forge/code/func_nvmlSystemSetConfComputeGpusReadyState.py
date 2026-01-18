from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlSystemSetConfComputeGpusReadyState(state):
    c_state = c_uint(state)
    fn = _nvmlGetFunctionPointer('nvmlSystemSetConfComputeGpusReadyState')
    ret = fn(c_state)
    _nvmlCheckReturn(ret)
    return ret