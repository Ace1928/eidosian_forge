from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceModifyDrainState(pciInfo, newState):
    fn = _nvmlGetFunctionPointer('nvmlDeviceModifyDrainState')
    ret = fn(pointer(pciInfo), newState)
    _nvmlCheckReturn(ret)
    return None