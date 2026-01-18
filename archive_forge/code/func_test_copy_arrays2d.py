import itertools
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests import usecases
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess
def test_copy_arrays2d(self):
    pyfunc = usecases.copy_arrays2d
    arraytype = types.Array(types.int32, 2, 'A')
    cfunc = njit((arraytype, arraytype))(pyfunc)
    nda = ((0, 0), (1, 1), (2, 5), (4, 25))
    for nd in nda:
        d1, d2 = nd
        a = np.arange(d1 * d2, dtype='int32').reshape(d1, d2)
        b = np.empty_like(a)
        args = (a, b)
        cfunc(*args)
        self.assertPreciseEqual(a, b, msg=str(args))