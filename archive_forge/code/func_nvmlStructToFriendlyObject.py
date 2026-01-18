from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlStructToFriendlyObject(struct):
    d = {}
    for x in struct._fields_:
        key = x[0]
        value = getattr(struct, key)
        d[key] = value.decode() if isinstance(value, bytes) else value
    obj = nvmlFriendlyObject(d)
    return obj