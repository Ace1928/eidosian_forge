import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def create_target_data(layout):
    """
    Create a TargetData instance for the given *layout* string.
    """
    return TargetData(ffi.lib.LLVMPY_CreateTargetData(_encode_string(layout)))