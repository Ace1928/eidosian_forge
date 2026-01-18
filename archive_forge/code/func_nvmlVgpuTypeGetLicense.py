from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlVgpuTypeGetLicense(vgpuTypeId):
    c_license = create_string_buffer(NVML_GRID_LICENSE_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_GRID_LICENSE_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetLicense')
    ret = fn(vgpuTypeId, c_license, c_buffer_size)
    _nvmlCheckReturn(ret)
    return c_license.value