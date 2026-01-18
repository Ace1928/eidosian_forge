import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def emit_assembly(self, module):
    """
        Return the raw assembler of the module, as a string.

        llvm.initialize_native_asmprinter() must have been called first.
        """
    return _decode_string(self._emit_to_memory(module, use_object=False))