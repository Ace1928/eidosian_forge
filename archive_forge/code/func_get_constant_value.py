from ctypes import (POINTER, byref, cast, c_char_p, c_double, c_int, c_size_t,
import enum
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.typeref import TypeRef
def get_constant_value(self, signed_int=False, round_fp=False):
    """
        Return the constant value, either as a literal (when supported)
        or as a string.

        Parameters
        -----------
        signed_int : bool
            if True and the constant is an integer, returns a signed version
        round_fp : bool
            if True and the constant is a floating point value, rounds the
            result upon accuracy loss (e.g., when querying an fp128 value).
            By default, raises an exception on accuracy loss
        """
    if not self.is_constant:
        raise ValueError('expected constant value, got %s' % (self._kind,))
    if self.value_kind == ValueKind.constant_int:
        little_endian = c_bool(False)
        words = ffi.lib.LLVMPY_GetConstantIntNumWords(self)
        ptr = ffi.lib.LLVMPY_GetConstantIntRawValue(self, byref(little_endian))
        asbytes = bytes(cast(ptr, POINTER(c_uint64 * words)).contents)
        return int.from_bytes(asbytes, 'little' if little_endian.value else 'big', signed=signed_int)
    elif self.value_kind == ValueKind.constant_fp:
        accuracy_loss = c_bool(False)
        value = ffi.lib.LLVMPY_GetConstantFPValue(self, byref(accuracy_loss))
        if accuracy_loss.value and (not round_fp):
            raise ValueError(f'Accuracy loss encountered in conversion of constant value {str(self)}')
        return value
    return str(self)