from ctypes import c_uint, c_bool
from llvmlite.binding import ffi
from llvmlite.binding import passmanagers
@inlining_threshold.setter
def inlining_threshold(self, threshold):
    ffi.lib.LLVMPY_PassManagerBuilderUseInlinerWithThreshold(self, threshold)