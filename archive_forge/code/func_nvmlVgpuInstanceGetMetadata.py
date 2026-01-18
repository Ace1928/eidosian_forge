from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetMetadata(vgpuInstance):
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetMetadata')
    c_vgpuMetadata = c_nvmlVgpuMetadata_t()
    c_bufferSize = c_uint(0)
    ret = fn(vgpuInstance, byref(c_vgpuMetadata), byref(c_bufferSize))
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        ret = fn(vgpuInstance, byref(c_vgpuMetadata), byref(c_bufferSize))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return c_vgpuMetadata