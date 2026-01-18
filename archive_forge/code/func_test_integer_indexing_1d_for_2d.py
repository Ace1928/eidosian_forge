import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_integer_indexing_1d_for_2d(self, flags=enable_pyobj_flags):
    pyfunc = integer_indexing_1d_usecase
    arraytype = types.Array(types.int32, 2, 'C')
    argtys = (arraytype, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(100, dtype='i4').reshape(10, 10)
    self.assertPreciseEqual(pyfunc(a, 0), cfunc(a, 0))
    self.assertPreciseEqual(pyfunc(a, 9), cfunc(a, 9))
    self.assertPreciseEqual(pyfunc(a, -1), cfunc(a, -1))
    arraytype = types.Array(types.int32, 2, 'A')
    argtys = (arraytype, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(20, dtype='i4').reshape(5, 4)[::2]
    self.assertPreciseEqual(pyfunc(a, 0), cfunc(a, 0))