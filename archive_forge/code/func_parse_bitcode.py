from ctypes import (c_char_p, byref, POINTER, c_bool, create_string_buffer,
from llvmlite.binding import ffi
from llvmlite.binding.linker import link_modules
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.value import ValueRef, TypeRef
from llvmlite.binding.context import get_global_context
def parse_bitcode(bitcode, context=None):
    """
    Create Module from a LLVM *bitcode* (a bytes object).
    """
    if context is None:
        context = get_global_context()
    buf = c_char_p(bitcode)
    bufsize = len(bitcode)
    with ffi.OutputString() as errmsg:
        mod = ModuleRef(ffi.lib.LLVMPY_ParseBitcode(context, buf, bufsize, errmsg), context)
        if errmsg:
            mod.close()
            raise RuntimeError('LLVM bitcode parsing error\n{0}'.format(errmsg))
    return mod