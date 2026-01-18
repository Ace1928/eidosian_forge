from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_licm_pass(self):
    """
        See http://llvm.org/docs/Passes.html#licm-loop-invariant-code-motion

        LLVM 14: `LLVMAddLICMPass`
        """
    ffi.lib.LLVMPY_AddLICMPass(self)