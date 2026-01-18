from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance):
    c_id = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetGpuInstanceId')
    ret = fn(vgpuInstance, byref(c_id))
    _nvmlCheckReturn(ret)
    return c_id.value