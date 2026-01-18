from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetVgpuSchedulerLog(handle):
    c_vgpu_sched_log = c_nvmlVgpuSchedulerLog_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuSchedulerLog')
    ret = fn(handle, byref(c_vgpu_sched_log))
    _nvmlCheckReturn(ret)
    return c_vgpu_sched_log