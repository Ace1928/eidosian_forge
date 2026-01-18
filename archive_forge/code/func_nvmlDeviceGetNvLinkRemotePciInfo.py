from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetNvLinkRemotePciInfo(device, link):
    c_pci = nvmlPciInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkRemotePciInfo_v2')
    ret = fn(device, link, byref(c_pci))
    _nvmlCheckReturn(ret)
    return c_pci