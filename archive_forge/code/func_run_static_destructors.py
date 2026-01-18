import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
def run_static_destructors(self):
    """
        Run static destructors which perform module-level cleanup of static
        resources.
        """
    ffi.lib.LLVMPY_RunStaticDestructors(self)