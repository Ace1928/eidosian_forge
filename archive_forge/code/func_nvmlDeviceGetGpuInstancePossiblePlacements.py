from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGpuInstancePossiblePlacements(device, profileId, placementsRef, countRef):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstancePossiblePlacements_v2')
    ret = fn(device, profileId, placementsRef, countRef)
    _nvmlCheckReturn(ret)
    return ret