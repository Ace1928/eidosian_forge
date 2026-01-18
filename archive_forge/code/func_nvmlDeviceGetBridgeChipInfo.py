from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetBridgeChipInfo(handle):
    bridgeHierarchy = c_nvmlBridgeChipHierarchy_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBridgeChipInfo')
    ret = fn(handle, byref(bridgeHierarchy))
    _nvmlCheckReturn(ret)
    return bridgeHierarchy