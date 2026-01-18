from llvmlite import ir
from numba.core import cgutils, types
from numba.core.extending import (intrinsic, make_attribute_wrapper, models,
from numba import float32
def next_double(bitgen):
    return bitgen.ctypes.next_double(bitgen.ctypes.state)