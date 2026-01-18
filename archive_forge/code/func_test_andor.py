import itertools
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests import usecases
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess
def test_andor(self):
    pyfunc = usecases.andor
    cfunc = njit((types.int32, types.int32))(pyfunc)
    xs = (-1, 0, 1, 9, 10, 11)
    ys = (-1, 0, 1, 9, 10, 11)
    for args in itertools.product(xs, ys):
        self.assertEqual(pyfunc(*args), cfunc(*args), 'args %s' % (args,))