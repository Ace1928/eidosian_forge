from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlComputeInstanceDestroy(computeInstance):
    fn = _nvmlGetFunctionPointer('nvmlComputeInstanceDestroy')
    ret = fn(computeInstance)
    _nvmlCheckReturn(ret)
    return ret