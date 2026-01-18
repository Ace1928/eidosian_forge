import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def test_loop_nest(self):
    """
        Test bug that decref the iterator early.
        If the bug occurs, a segfault should occur
        """
    pyfunc = loop_nest_3
    cfunc = jit((), forceobj=True)(pyfunc)
    self.assertEqual(pyfunc(5, 5), cfunc(5, 5))

    def bm_pyfunc():
        pyfunc(5, 5)

    def bm_cfunc():
        cfunc(5, 5)
    utils.benchmark(bm_pyfunc)
    utils.benchmark(bm_cfunc)