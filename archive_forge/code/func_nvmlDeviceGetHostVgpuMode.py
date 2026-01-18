from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetHostVgpuMode(handle):
    c_host_vgpu_mode = _nvmlHostVgpuMode_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetHostVgpuMode')
    ret = fn(handle, byref(c_host_vgpu_mode))
    _nvmlCheckReturn(ret)
    return c_host_vgpu_mode.value