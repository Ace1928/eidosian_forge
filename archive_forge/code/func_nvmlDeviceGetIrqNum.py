from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetIrqNum(device):
    c_irqNum = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetIrqNum')
    ret = fn(device, byref(c_irqNum))
    _nvmlCheckReturn(ret)
    return c_irqNum.value