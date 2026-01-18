from ctypes import byref, c_char, c_char_p, c_int, c_size_t, c_void_p, POINTER
from enum import IntEnum
from numba.core import config
from numba.cuda.cudadrv.error import (NvrtcError, NvrtcCompilationError,
import functools
import os
import threading
import warnings
def compile_program(self, program, options):
    """
        Compile an NVRTC program. Compilation may fail due to a user error in
        the source; this function returns ``True`` if there is a compilation
        error and ``False`` on success.
        """
    encoded_options = [opt.encode() for opt in options]
    option_pointers = [c_char_p(opt) for opt in encoded_options]
    c_options_type = c_char_p * len(options)
    c_options = c_options_type(*option_pointers)
    try:
        self.nvrtcCompileProgram(program.handle, len(options), c_options)
        return False
    except NvrtcCompilationError:
        return True