from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlVgpuInstanceGetMdevUUID(vgpuInstance):
    c_uuid = create_string_buffer(NVML_DEVICE_UUID_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_DEVICE_UUID_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetMdevUUID')
    ret = fn(vgpuInstance, byref(c_uuid), c_buffer_size)
    _nvmlCheckReturn(ret)
    return c_uuid.value