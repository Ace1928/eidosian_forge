import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_all_args(self):
    """
        Tests calling a function when n_args > 30 and
        n_kws = 0. This shouldn't use the peephole, but
        it should still succeed.
        """
    total_args = [i for i in range(self.THRESHOLD_ARGS)]
    f = self.gen_func(self.THRESHOLD_ARGS, 0)
    py_func = f
    cfunc = njit()(f)
    a = py_func(*total_args)
    b = cfunc(*total_args)
    self.assertEqual(a, b)