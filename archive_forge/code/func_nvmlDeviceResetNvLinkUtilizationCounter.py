from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter):
    fn = _nvmlGetFunctionPointer('nvmlDeviceResetNvLinkUtilizationCounter')
    ret = fn(device, link, counter)
    _nvmlCheckReturn(ret)
    return None