from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlComputeInstanceGetInfo(computeInstance):
    return nvmlComputeInstanceGetInfo_v2(computeInstance)