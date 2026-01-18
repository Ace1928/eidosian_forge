import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_large_args_small_kws(self):
    """
        Tests calling a function when (n_args / 2) + n_kws > 15,
        but n_args > 30 and n_kws <= 15
        """
    used_args = self.THRESHOLD_ARGS
    used_kws = self.THRESHOLD_KWS - 1
    total_args = [i for i in range(used_args + used_kws)]
    f = self.gen_func(used_args, used_kws)
    py_func = f
    cfunc = njit()(f)
    a = py_func(*total_args)
    b = cfunc(*total_args)
    self.assertEqual(a, b)