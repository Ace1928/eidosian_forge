import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
def get_function_address(self, name):
    """
        Return the address of the function named *name* as an integer.

        It's a fatal error in LLVM if the symbol of *name* doesn't exist.
        """
    return ffi.lib.LLVMPY_GetFunctionAddress(self, name.encode('ascii'))