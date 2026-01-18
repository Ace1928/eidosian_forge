from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlDeviceGetBoardPartNumber(handle):
    c_part_number = create_string_buffer(NVML_DEVICE_PART_NUMBER_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBoardPartNumber')
    ret = fn(handle, c_part_number, c_uint(NVML_DEVICE_PART_NUMBER_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_part_number.value