import itertools
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests import usecases
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess
def test_copy_arrays(self):
    pyfunc = usecases.copy_arrays
    arraytype = types.Array(types.int32, 1, 'A')
    cfunc = njit((arraytype, arraytype))(pyfunc)
    nda = (0, 1, 10, 100)
    for nd in nda:
        a = np.arange(nd, dtype='int32')
        b = np.empty_like(a)
        args = (a, b)
        cfunc(*args)
        self.assertPreciseEqual(a, b, msg=str(args))