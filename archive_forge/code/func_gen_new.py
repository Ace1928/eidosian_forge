from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def gen_new(val):

    def new(typ):
        obj = NVMLError.__new__(typ, val)
        return obj
    return new