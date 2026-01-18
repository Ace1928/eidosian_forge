from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance):
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetLicenseInfo_v2')
    c_license_info = c_nvmlVgpuLicenseInfo_t()
    ret = fn(vgpuInstance, byref(c_license_info))
    _nvmlCheckReturn(ret)
    return c_license_info