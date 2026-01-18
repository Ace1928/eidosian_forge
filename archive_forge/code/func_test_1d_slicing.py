import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_1d_slicing(self, flags=enable_pyobj_flags):
    pyfunc = slicing_1d_usecase
    arraytype = types.Array(types.int32, 1, 'C')
    argtys = (arraytype, types.int32, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(10, dtype='i4')
    for indices in [(0, 10, 1), (2, 3, 1), (10, 0, 1), (0, 10, -1), (0, 10, 2), (9, 0, -1), (-5, -2, 1), (0, -1, 1)]:
        expected = pyfunc(a, *indices)
        self.assertPreciseEqual(cfunc(a, *indices), expected)