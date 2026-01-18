import contextlib
import ctypes
from ctypes import (
from ctypes.util import find_library
def create_cfbool(b):
    return CFNumberCreate(None, 9, ctypes.byref(c_int32(1 if b else 0)))