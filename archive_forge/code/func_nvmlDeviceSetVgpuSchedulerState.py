from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetVgpuSchedulerState(handle, sched_state):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetVgpuSchedulerState')
    ret = fn(handle, byref(sched_state))
    _nvmlCheckReturn(ret)
    return ret