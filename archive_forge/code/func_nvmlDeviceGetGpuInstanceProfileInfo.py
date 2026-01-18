from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGpuInstanceProfileInfo(device, profile, version=2):
    if version == 2:
        c_info = c_nvmlGpuInstanceProfileInfo_v2_t()
        fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstanceProfileInfoV')
    elif version == 1:
        c_info = c_nvmlGpuInstanceProfileInfo_t()
        fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstanceProfileInfo')
    else:
        raise NVMLError(NVML_ERROR_FUNCTION_NOT_FOUND)
    ret = fn(device, profile, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info