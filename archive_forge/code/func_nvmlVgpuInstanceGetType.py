from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetType(vgpuInstance):
    c_vgpu_type = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetType')
    ret = fn(vgpuInstance, byref(c_vgpu_type))
    _nvmlCheckReturn(ret)
    return c_vgpu_type.value