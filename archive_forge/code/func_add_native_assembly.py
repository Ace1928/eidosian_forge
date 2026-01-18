import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
def add_native_assembly(self, asm):
    """
        Adds a compilation unit to the library using native assembly as the
        input format.

        This takes a string or an object that can be converted to a string that
        contains native assembly, which will be
        parsed by LLVM.
        """
    self.__entries.append((1, str(asm).encode('utf-8')))
    return self