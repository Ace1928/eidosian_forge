from llvmlite import ir
from numba.core import cgutils, types
from numba.core.extending import (intrinsic, make_attribute_wrapper, models,
from numba import float32
def next_uint32(bitgen):
    return bitgen.ctypes.next_uint32(bitgen.ctypes.state)