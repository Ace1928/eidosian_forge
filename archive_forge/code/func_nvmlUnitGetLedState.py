from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlUnitGetLedState(unit):
    c_state = c_nvmlLedState_t()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetLedState')
    ret = fn(unit, byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state