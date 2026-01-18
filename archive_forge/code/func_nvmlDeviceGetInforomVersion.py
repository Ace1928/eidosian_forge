from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlDeviceGetInforomVersion(handle, infoRomObject):
    c_version = create_string_buffer(NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetInforomVersion')
    ret = fn(handle, _nvmlInforomObject_t(infoRomObject), c_version, c_uint(NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value