from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetConfComputeGpuCertificate(device):
    c_cert = c_nvmlConfComputeGpuCertificate_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetConfComputeGpuCertificate')
    ret = fn(device, byref(c_cert))
    _nvmlCheckReturn(ret)
    return c_cert