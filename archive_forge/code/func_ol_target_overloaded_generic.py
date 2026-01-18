from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
@overload(target_overloaded, target='generic')
def ol_target_overloaded_generic(x):

    def impl(x):
        x[0] *= GENERIC_TARGET_OL
    return impl