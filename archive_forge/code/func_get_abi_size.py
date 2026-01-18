import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def get_abi_size(self, ty):
    """
        Get ABI size of LLVM type *ty*.
        """
    return ffi.lib.LLVMPY_ABISizeOfType(self, ty)