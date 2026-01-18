from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlSystemSetNvlinkBwMode(mode):
    fn = _nvmlGetFunctionPointer('nvmlSystemSetNvlinkBwMode')
    ret = fn(mode)
    _nvmlCheckReturn(ret)
    return ret