from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGridLicensableFeatures_v4(handle):
    c_get_grid_licensable_features = c_nvmlGridLicensableFeatures_v4_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGridLicensableFeatures_v4')
    ret = fn(handle, byref(c_get_grid_licensable_features))
    _nvmlCheckReturn(ret)
    return c_get_grid_licensable_features