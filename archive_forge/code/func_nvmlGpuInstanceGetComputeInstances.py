from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, computeInstancesRef, countRef):
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetComputeInstances')
    ret = fn(gpuInstance, profileId, computeInstancesRef, countRef)
    _nvmlCheckReturn(ret)
    return ret