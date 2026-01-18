from ctypes import c_uint, c_bool
from llvmlite.binding import ffi
from llvmlite.binding import passmanagers
@loop_vectorize.setter
def loop_vectorize(self, enable=True):
    return ffi.lib.LLVMPY_PassManagerBuilderSetLoopVectorize(self, enable)