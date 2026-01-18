from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlEventSetWait(eventSet, timeoutms):
    return nvmlEventSetWait_v2(eventSet, timeoutms)