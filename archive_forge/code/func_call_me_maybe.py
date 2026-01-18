from ctypes import *
import sys
import numpy as np
from numba import _helperlib
def call_me_maybe(arr):
    return mydct[arr[0].decode('ascii')]