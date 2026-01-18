from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlSystemGetNVMLVersion():
    c_version = create_string_buffer(NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlSystemGetNVMLVersion')
    ret = fn(c_version, c_uint(NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value