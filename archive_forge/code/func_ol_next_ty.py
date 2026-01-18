from llvmlite import ir
from numba.core import cgutils, types
from numba.core.extending import (intrinsic, make_attribute_wrapper, models,
from numba import float32
@overload(overloadable_function)
def ol_next_ty(bitgen):
    if isinstance(bitgen, types.NumPyRandomBitGeneratorType):

        def impl(bitgen):
            return intrin_NumPyRandomBitGeneratorType_next_ty(bitgen)
        return impl