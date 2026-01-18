from ctypes import c_int, c_bool, c_void_p, c_uint64
import enum
from llvmlite.binding import ffi
@property
def is_function_vararg(self):
    """
        Returns true if a function type accepts a variable number of arguments.
        When the type is not a function, raises exception.
        """
    if self.type_kind != TypeKind.function:
        raise ValueError('Type {} is not a function'.format(self))
    return ffi.lib.LLVMPY_IsFunctionVararg(self)