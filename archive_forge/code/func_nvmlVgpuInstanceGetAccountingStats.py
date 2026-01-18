from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid):
    c_accountingStats = c_nvmlAccountingStats_t()
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetAccountingStats')
    ret = fn(vgpuInstance, pid, byref(c_accountingStats))
    _nvmlCheckReturn(ret)
    return c_accountingStats