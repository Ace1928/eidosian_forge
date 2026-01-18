from ctypes import c_uint
from llvmlite.binding import ffi
def initialize_native_asmparser():
    """
    Initialize the native ASM parser.
    """
    ffi.lib.LLVMPY_InitializeNativeAsmParser()