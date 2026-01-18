from ctypes import (POINTER, byref, cast, c_char_p, c_double, c_int, c_size_t,
import enum
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.typeref import TypeRef
@property
def operands(self):
    """
        Return an iterator over this instruction's operands.
        The iterator will yield a ValueRef for each operand.
        """
    if not self.is_instruction:
        raise ValueError('expected instruction value, got %s' % (self._kind,))
    it = ffi.lib.LLVMPY_InstructionOperandsIter(self)
    parents = self._parents.copy()
    parents.update(instruction=self)
    return _OperandsIterator(it, parents)