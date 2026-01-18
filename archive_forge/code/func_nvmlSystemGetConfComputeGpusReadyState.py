from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlSystemGetConfComputeGpusReadyState():
    c_state = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlSystemGetConfComputeGpusReadyState')
    ret = fn(byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state.value