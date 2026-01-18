from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetCurrentEccMode(handle):
    return nvmlDeviceGetEccMode(handle)[0]