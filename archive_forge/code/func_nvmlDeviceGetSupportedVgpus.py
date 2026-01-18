from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetSupportedVgpus(handle):
    c_vgpu_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSupportedVgpus')
    ret = fn(handle, byref(c_vgpu_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        vgpu_type_ids_array = _nvmlVgpuTypeId_t * c_vgpu_count.value
        c_vgpu_type_ids = vgpu_type_ids_array()
        ret = fn(handle, byref(c_vgpu_count), c_vgpu_type_ids)
        _nvmlCheckReturn(ret)
        vgpus = []
        for i in range(c_vgpu_count.value):
            vgpus.append(c_vgpu_type_ids[i])
        return vgpus
    else:
        raise NVMLError(ret)