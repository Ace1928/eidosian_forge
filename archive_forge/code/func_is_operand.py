from ctypes import (POINTER, byref, cast, c_char_p, c_double, c_int, c_size_t,
import enum
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.typeref import TypeRef
@property
def is_operand(self):
    return self._kind == 'operand'