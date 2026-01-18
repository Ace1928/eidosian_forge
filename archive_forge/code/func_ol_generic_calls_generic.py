from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
@overload(generic_calls_generic, target='generic')
def ol_generic_calls_generic(x):

    def impl(x):
        x[0] *= GENERIC_CALLS_GENERIC
        generic_func_1(x)
    return impl