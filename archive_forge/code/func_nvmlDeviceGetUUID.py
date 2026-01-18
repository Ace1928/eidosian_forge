from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlDeviceGetUUID(handle):
    c_uuid = create_string_buffer(NVML_DEVICE_UUID_V2_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetUUID')
    ret = fn(handle, c_uuid, c_uint(NVML_DEVICE_UUID_V2_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_uuid.value