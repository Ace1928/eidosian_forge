from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpuInstanceCreateComputeInstanceWithPlacement(gpuInstance, profileId, placement):
    c_instance = c_nvmlComputeInstance_t()
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceCreateComputeInstanceWithPlacement')
    ret = fn(gpuInstance, profileId, placement, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance