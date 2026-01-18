from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMemoryAffinity(handle, nodeSetSize, scope):
    affinity_array = c_ulonglong * nodeSetSize
    c_affinity = affinity_array()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemoryAffinity')
    ret = fn(handle, nodeSetSize, byref(c_affinity), _nvmlAffinityScope_t(scope))
    _nvmlCheckReturn(ret)
    return c_affinity