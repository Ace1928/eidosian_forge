from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGraphicsRunningProcesses(handle):
    return nvmlDeviceGetGraphicsRunningProcesses_v3(handle)