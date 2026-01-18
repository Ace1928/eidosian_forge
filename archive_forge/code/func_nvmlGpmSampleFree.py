from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpmSampleFree(gpmSample):
    fn = _nvmlGetFunctionPointer('nvmlGpmSampleFree')
    ret = fn(gpmSample)
    _nvmlCheckReturn(ret)
    return