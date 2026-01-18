from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpmSampleGet(device, gpmSample):
    fn = _nvmlGetFunctionPointer('nvmlGpmSampleGet')
    ret = fn(device, gpmSample)
    _nvmlCheckReturn(ret)
    return gpmSample