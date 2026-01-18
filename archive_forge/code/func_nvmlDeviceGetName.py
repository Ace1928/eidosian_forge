from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlDeviceGetName(handle):
    c_name = create_string_buffer(NVML_DEVICE_NAME_V2_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetName')
    ret = fn(handle, c_name, c_uint(NVML_DEVICE_NAME_V2_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_name.value