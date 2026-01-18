from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGpuFabricInfo(device, gpuFabricInfo):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuFabricInfo')
    ret = fn(device, gpuFabricInfo)
    _nvmlCheckReturn(ret)
    return ret