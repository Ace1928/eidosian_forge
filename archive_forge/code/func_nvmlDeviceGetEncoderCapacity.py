from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetEncoderCapacity(handle, encoderQueryType):
    c_encoder_capacity = c_ulonglong(0)
    c_encoderQuery_type = _nvmlEncoderQueryType_t(encoderQueryType)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetEncoderCapacity')
    ret = fn(handle, c_encoderQuery_type, byref(c_encoder_capacity))
    _nvmlCheckReturn(ret)
    return c_encoder_capacity.value