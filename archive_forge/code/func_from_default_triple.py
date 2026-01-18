import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
@classmethod
def from_default_triple(cls):
    """
        Create a Target instance for the default triple.
        """
    triple = get_default_triple()
    return cls.from_triple(triple)