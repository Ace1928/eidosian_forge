from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpmQueryDeviceSupport(device):
    gpmSupport = c_nvmlGpmSupport_t()
    gpmSupport.version = NVML_GPM_SUPPORT_VERSION
    fn = _nvmlGetFunctionPointer('nvmlGpmQueryDeviceSupport')
    ret = fn(device, byref(gpmSupport))
    _nvmlCheckReturn(ret)
    return gpmSupport