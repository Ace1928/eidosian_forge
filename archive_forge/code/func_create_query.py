import contextlib
import ctypes
from ctypes import (
from ctypes.util import find_library
def create_query(**kwargs):
    return CFDictionaryCreate(None, (c_void_p * len(kwargs))(*[k_(k) for k in kwargs.keys()]), (c_void_p * len(kwargs))(*[create_cfstr(v) if isinstance(v, str) else v for v in kwargs.values()]), len(kwargs), _found.kCFTypeDictionaryKeyCallBacks, _found.kCFTypeDictionaryValueCallBacks)