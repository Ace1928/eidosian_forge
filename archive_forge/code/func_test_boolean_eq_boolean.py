import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_boolean_eq_boolean(self):
    pyfunc = equality
    cfunc = njit((types.boolean, types.boolean))(pyfunc)
    xs = (True, False)
    ys = (True, False)
    for xs, ys in itertools.product(xs, ys):
        self.assertEqual(pyfunc(xs, ys), cfunc(xs, ys))