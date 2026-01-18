from types import BuiltinFunctionType
import ctypes
from functools import partial
import numpy as np
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.np import numpy_support
def is_ffi_instance(obj):
    try:
        return obj in _ffi_instances or isinstance(obj, cffi.FFI)
    except TypeError:
        return False