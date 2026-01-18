from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpuInstanceGetComputeInstanceById(gpuInstance, computeInstanceId):
    c_instance = c_nvmlComputeInstance_t()
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetComputeInstanceById')
    ret = fn(gpuInstance, computeInstanceId, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance