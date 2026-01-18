import ctypes
import sys
from numba.core import types
from numba.core.typing import templates
from .typeof import typeof_impl
def is_ctypes_funcptr(obj):
    try:
        ctypes.cast(obj, ctypes.c_void_p)
    except ctypes.ArgumentError:
        return False
    else:
        return hasattr(obj, 'argtypes') and hasattr(obj, 'restype')