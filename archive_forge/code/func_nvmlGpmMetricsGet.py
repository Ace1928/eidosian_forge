from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpmMetricsGet(metricsGet):
    fn = _nvmlGetFunctionPointer('nvmlGpmMetricsGet')
    ret = fn(byref(metricsGet))
    _nvmlCheckReturn(ret)
    return metricsGet