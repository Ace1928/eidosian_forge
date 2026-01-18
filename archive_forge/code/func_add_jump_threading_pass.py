from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_jump_threading_pass(self, threshold=-1):
    """
        See https://llvm.org/docs/Passes.html#jump-threading-jump-threading

        LLVM 14: `llvm::createJumpThreadingPass`
        """
    ffi.lib.LLVMPY_AddJumpThreadingPass(self, threshold)