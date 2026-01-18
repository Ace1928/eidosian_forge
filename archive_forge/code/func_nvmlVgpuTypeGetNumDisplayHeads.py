from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId):
    c_num_heads = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetNumDisplayHeads')
    ret = fn(vgpuTypeId, byref(c_num_heads))
    _nvmlCheckReturn(ret)
    return c_num_heads.value