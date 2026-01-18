from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlSystemGetNvlinkBwMode():
    mode = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlSystemGetNvlinkBwMode')
    ret = fn(byref(mode))
    _nvmlCheckReturn(ret)
    return mode.value