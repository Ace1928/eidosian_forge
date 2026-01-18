from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_single_loop_extractor_pass(self):
    """
        See https://llvm.org/docs/Passes.html#loop-extract-single-extract-at-most-one-loop-into-a-new-function

        LLVM 14: `llvm::createSingleLoopExtractorPass`
        """
    ffi.lib.LLVMPY_AddSingleLoopExtractorPass(self)