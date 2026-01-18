from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_reverse_post_order_function_attrs_pass(self):
    """
        See https://llvm.org/docs/Passes.html#function-attrs-deduce-function-attributes

        LLVM 14: `llvm::createReversePostOrderFunctionAttrsPass`
        """
    ffi.lib.LLVMPY_AddReversePostOrderFunctionAttrsPass(self)