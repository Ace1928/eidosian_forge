from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlSystemGetConfComputeState():
    c_state = c_nvmlConfComputeSystemState_t()
    fn = _nvmlGetFunctionPointer('nvmlSystemGetConfComputeState')
    ret = fn(byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state