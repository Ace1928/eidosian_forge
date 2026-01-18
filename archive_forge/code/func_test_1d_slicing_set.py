import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_1d_slicing_set(self, flags=enable_pyobj_flags):
    """
        1d to 1d slice assignment
        """
    pyfunc = slicing_1d_usecase_set
    dest_type = types.Array(types.int32, 1, 'C')
    src_type = types.Array(types.int16, 1, 'A')
    argtys = (dest_type, src_type, types.int32, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    N = 10
    arg = np.arange(N, dtype='i2') + 40
    bounds = [0, 2, N - 2, N, N + 1, N + 3, -2, -N + 2, -N, -N - 1, -N - 3]

    def make_dest():
        return np.zeros_like(arg, dtype='i4')
    for start, stop in itertools.product(bounds, bounds):
        for step in (1, 2, -1, -2):
            args = (start, stop, step)
            index = slice(*args)
            pyleft = pyfunc(make_dest(), arg[index], *args)
            cleft = cfunc(make_dest(), arg[index], *args)
            self.assertPreciseEqual(pyleft, cleft)
    with self.assertRaises(ValueError):
        cfunc(np.zeros_like(arg, dtype=np.int32), arg, 0, 0, 1)