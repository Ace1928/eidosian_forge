import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
def run_static_constructors(self):
    """
        Run static constructors which initialize module-level static objects.
        """
    ffi.lib.LLVMPY_RunStaticConstructors(self)