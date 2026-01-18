from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetAPIRestriction(handle, apiType, isRestricted):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetAPIRestriction')
    ret = fn(handle, _nvmlRestrictedAPI_t(apiType), _nvmlEnableState_t(isRestricted))
    _nvmlCheckReturn(ret)
    return None