from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetVirtualizationMode(handle, virtualization_mode):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetVirtualizationMode')
    return fn(handle, virtualization_mode)