from ctypes import c_uint, c_bool
from llvmlite.binding import ffi
from llvmlite.binding import passmanagers
@opt_level.setter
def opt_level(self, level):
    ffi.lib.LLVMPY_PassManagerBuilderSetOptLevel(self, level)