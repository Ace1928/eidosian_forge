from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetTopologyNearestGpus(device, level):
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTopologyNearestGpus')
    ret = fn(device, level, byref(c_count), None)
    if ret != NVML_SUCCESS:
        raise NVMLError(ret)
    device_array = c_nvmlDevice_t * c_count.value
    c_devices = device_array()
    ret = fn(device, level, byref(c_count), c_devices)
    _nvmlCheckReturn(ret)
    return list(c_devices[0:c_count.value])