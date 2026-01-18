from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetP2PStatus(device1, device2, p2pIndex):
    c_p2pstatus = _nvmlGpuP2PStatus_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetP2PStatus')
    ret = fn(device1, device2, p2pIndex, byref(c_p2pstatus))
    _nvmlCheckReturn(ret)
    return c_p2pstatus.value