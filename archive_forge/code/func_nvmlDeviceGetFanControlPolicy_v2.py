from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetFanControlPolicy_v2(handle, fan, fanControlPolicy):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetFanControlPolicy_v2')
    ret = fn(handle, fan, fanControlPolicy)
    _nvmlCheckReturn(ret)
    return ret