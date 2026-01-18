from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlSystemGetCudaDriverVersion():
    c_cuda_version = c_int()
    fn = _nvmlGetFunctionPointer('nvmlSystemGetCudaDriverVersion')
    ret = fn(byref(c_cuda_version))
    _nvmlCheckReturn(ret)
    return c_cuda_version.value