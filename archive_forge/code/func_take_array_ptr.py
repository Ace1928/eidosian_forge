from ctypes import *
import sys
import numpy as np
from numba import _helperlib
def take_array_ptr(ptr):
    return ptr