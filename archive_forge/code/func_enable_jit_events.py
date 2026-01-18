import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
def enable_jit_events(self):
    """
        Enable JIT events for profiling of generated code.
        Return value indicates whether connection to profiling tool
        was successful.
        """
    ret = ffi.lib.LLVMPY_EnableJITEvents(self)
    return ret