from ctypes import (c_char_p, byref, POINTER, c_bool, create_string_buffer,
from llvmlite.binding import ffi
from llvmlite.binding.linker import link_modules
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.value import ValueRef, TypeRef
from llvmlite.binding.context import get_global_context
@property
def struct_types(self):
    """
        Return an iterator over the struct types defined in
        the module. The iterator will yield a TypeRef.
        """
    it = ffi.lib.LLVMPY_ModuleTypesIter(self)
    return _TypesIterator(it, dict(module=self))