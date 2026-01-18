from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_instruction_namer_pass(self):
    """
        See https://llvm.org/docs/Passes.html#instnamer-assign-names-to-anonymous-instructions.

        LLVM 14: `llvm::createInstructionNamerPass`
        """
    ffi.lib.LLVMPY_AddInstructionNamerPass(self)