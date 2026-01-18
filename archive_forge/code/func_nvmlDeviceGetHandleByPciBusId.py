from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlDeviceGetHandleByPciBusId(pciBusId):
    c_busId = c_char_p(pciBusId)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetHandleByPciBusId_v2')
    ret = fn(c_busId, byref(device))
    _nvmlCheckReturn(ret)
    return device