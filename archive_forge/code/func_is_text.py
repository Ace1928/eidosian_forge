from llvmlite.binding import ffi
from ctypes import (c_bool, c_char_p, c_char, c_size_t, string_at, c_uint64,
def is_text(self):
    return ffi.lib.LLVMPY_IsSectionText(self)