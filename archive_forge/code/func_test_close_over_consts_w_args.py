from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
def test_close_over_consts_w_args(self):

    def impl_factory(consumer_func):

        def impl(x):
            y = 10

            def callinner(z):
                return y + z + _global
            return consumer_func(callinner, x)
        return impl
    cfunc = njit(impl_factory(consumer))
    impl = impl_factory(consumer.py_func)
    a = 5
    self.assertEqual(impl(a), cfunc(a))