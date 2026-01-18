from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceClearAccountingPids(vgpuInstance):
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceClearAccountingPids')
    ret = fn(vgpuInstance)
    _nvmlCheckReturn(ret)
    return ret