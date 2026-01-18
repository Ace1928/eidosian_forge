from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetAttributes(device):
    return nvmlDeviceGetAttributes_v2(device)