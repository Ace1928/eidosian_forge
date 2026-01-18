from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlVgpuInstanceGetGpuPciId(vgpuInstance):
    c_vgpuPciId = create_string_buffer(NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetGpuPciId')
    ret = fn(vgpuInstance, c_vgpuPciId, byref(c_uint(NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE)))
    _nvmlCheckReturn(ret)
    return c_vgpuPciId.value