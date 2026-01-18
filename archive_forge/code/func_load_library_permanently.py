from ctypes import c_void_p, c_char_p, c_bool, POINTER
from llvmlite.binding import ffi
from llvmlite.binding.common import _encode_string
def load_library_permanently(filename):
    """
    Load an external library
    """
    with ffi.OutputString() as outerr:
        if ffi.lib.LLVMPY_LoadLibraryPermanently(_encode_string(filename), outerr):
            raise RuntimeError(str(outerr))