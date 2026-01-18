from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_lcssa_pass(self):
    """
        See https://llvm.org/docs/Passes.html#lcssa-loop-closed-ssa-form-pass

        LLVM 14: `llvm::createLCSSAPass`
        """
    ffi.lib.LLVMPY_AddLCSSAPass(self)