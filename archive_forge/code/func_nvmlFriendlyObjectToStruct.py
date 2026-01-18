from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlFriendlyObjectToStruct(obj, model):
    for x in model._fields_:
        key = x[0]
        value = obj.__dict__[key]
        if sys.version_info >= (3,):
            setattr(model, key, value.encode())
        else:
            setattr(model, key, value)
    return model