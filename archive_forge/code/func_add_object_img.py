import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
def add_object_img(self, data):
    """
        Adds a compilation unit to the library using pre-compiled object code.

        This takes the bytes of the contents of an object artifact which will be
        loaded by LLVM.
        """
    self.__entries.append((2, bytes(data)))
    return self