from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
@unittest.skip('Needs option/flag inheritance to work')
def test_jit_option_inheritance(self):

    def impl_factory(consumer_func):

        def impl(x):

            def inner(val):
                return 1 / val
            return consumer_func(inner, x)
        return impl
    cfunc = njit(error_model='numpy')(impl_factory(consumer))
    impl = impl_factory(consumer.py_func)
    a = 0
    self.assertEqual(impl(a), cfunc(a))