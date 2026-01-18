from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_strip_dead_debug_info_pass(self):
    """
        See https://llvm.org/docs/Passes.html#strip-dead-debug-info-strip-debug-info-for-unused-symbols

        LLVM 14: `llvm::createStripDeadDebugInfoPass`
        """
    ffi.lib.LLVMPY_AddStripDeadDebugInfoPass(self)