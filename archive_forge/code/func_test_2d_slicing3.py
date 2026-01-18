import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_2d_slicing3(self, flags=enable_pyobj_flags):
    """
        arr_2d[a:b:c, d]
        """
    pyfunc = slicing_2d_usecase3
    arraytype = types.Array(types.int32, 2, 'C')
    argtys = (arraytype, types.int32, types.int32, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(100, dtype='i4').reshape(10, 10)
    args = [(0, 10, 1, 0), (2, 3, 1, 1), (10, 0, -1, 8), (9, 0, -2, 4), (0, 10, 2, 3), (0, -1, 3, 1)]
    for arg in args:
        expected = pyfunc(a, *arg)
        self.assertPreciseEqual(cfunc(a, *arg), expected)
    arraytype = types.Array(types.int32, 2, 'A')
    argtys = (arraytype, types.int32, types.int32, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(400, dtype='i4').reshape(20, 20)[::2, ::2]
    for arg in args:
        expected = pyfunc(a, *arg)
        self.assertPreciseEqual(cfunc(a, *arg), expected)