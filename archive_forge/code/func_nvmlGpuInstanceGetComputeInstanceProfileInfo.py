from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpuInstanceGetComputeInstanceProfileInfo(device, profile, engProfile, version=2):
    if version == 2:
        c_info = c_nvmlComputeInstanceProfileInfo_v2_t()
        fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetComputeInstanceProfileInfoV')
    elif version == 1:
        c_info = c_nvmlComputeInstanceProfileInfo_t()
        fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetComputeInstanceProfileInfo')
    else:
        raise NVMLError(NVML_ERROR_FUNCTION_NOT_FOUND)
    ret = fn(device, profile, engProfile, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info