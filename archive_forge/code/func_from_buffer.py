import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
def from_buffer(obj):
    """Return ``(pointer_address, length_in_bytes)`` for a buffer object."""
    if hasattr(obj, 'buffer_info'):
        address, length = obj.buffer_info()
        return (address, length * obj.itemsize)
    elif hasattr(obj, '__array_interface__'):
        length = reduce(operator.mul, obj.shape)
        return (ctypes.addressof(ctypes.c_char.from_buffer(obj)), length)
    else:
        return (ctypes.addressof(ctypes.c_char.from_buffer(obj)), len(obj))