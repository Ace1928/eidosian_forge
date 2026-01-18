import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_boolean_as_int(self):
    pyfunc = equality
    cfunc = njit((types.boolean, types.intp))(pyfunc)
    xs = (True, False)
    ys = (-1, 0, 1)
    for xs, ys in itertools.product(xs, ys):
        self.assertEqual(pyfunc(xs, ys), cfunc(xs, ys))