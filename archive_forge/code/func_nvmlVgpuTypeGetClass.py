from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlVgpuTypeGetClass(vgpuTypeId):
    c_class = create_string_buffer(NVML_DEVICE_NAME_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_DEVICE_NAME_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetClass')
    ret = fn(vgpuTypeId, c_class, byref(c_buffer_size))
    _nvmlCheckReturn(ret)
    return c_class.value