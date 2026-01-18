import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def check_ellipsis(self, pyfunc, flags):

    def compile_func(arr):
        argtys = (typeof(arr), types.intp, types.intp)
        return jit(argtys, **flags)(pyfunc)

    def run(a):
        bounds = (0, 1, 2, -1, -2)
        cfunc = compile_func(a)
        for i, j in itertools.product(bounds, bounds):
            x = cfunc(a, i, j)
            np.testing.assert_equal(pyfunc(a, i, j), cfunc(a, i, j))
    run(np.arange(16, dtype='i4').reshape(4, 4))
    run(np.arange(27, dtype='i4').reshape(3, 3, 3))