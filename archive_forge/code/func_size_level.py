from ctypes import c_uint, c_bool
from llvmlite.binding import ffi
from llvmlite.binding import passmanagers
@size_level.setter
def size_level(self, size):
    ffi.lib.LLVMPY_PassManagerBuilderSetSizeLevel(self, size)