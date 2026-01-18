from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetFanControlPolicy(handle, fan, fanControlPolicy):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetFanControlPolicy')
    ret = fn(handle, fan, _nvmlFanControlPolicy_t(fanControlPolicy))
    _nvmlCheckReturn(ret)
    return ret