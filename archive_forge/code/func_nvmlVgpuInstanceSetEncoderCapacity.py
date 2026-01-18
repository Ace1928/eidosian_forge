from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoder_capacity):
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceSetEncoderCapacity')
    return fn(vgpuInstance, encoder_capacity)