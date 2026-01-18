from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpuInstanceDestroy(gpuInstance):
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceDestroy')
    ret = fn(gpuInstance)
    _nvmlCheckReturn(ret)
    return ret