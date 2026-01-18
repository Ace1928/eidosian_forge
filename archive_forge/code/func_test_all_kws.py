import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_all_kws(self):
    """
        Tests calling a function when n_kws > 15 and
        n_args = 0.
        """
    total_args = [i for i in range(self.THRESHOLD_KWS)]
    f = self.gen_func(0, self.THRESHOLD_KWS)
    py_func = f
    cfunc = njit()(f)
    a = py_func(*total_args)
    b = cfunc(*total_args)
    self.assertEqual(a, b)