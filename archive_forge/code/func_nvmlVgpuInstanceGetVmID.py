from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlVgpuInstanceGetVmID(vgpuInstance):
    c_vm_id = create_string_buffer(NVML_DEVICE_UUID_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_GRID_LICENSE_BUFFER_SIZE)
    c_vm_id_type = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetVmID')
    ret = fn(vgpuInstance, byref(c_vm_id), c_buffer_size, byref(c_vm_id_type))
    _nvmlCheckReturn(ret)
    return (c_vm_id.value, c_vm_id_type.value)