from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
def test_nested_escape(self):

    def impl_factory(consumer_func):

        def impl():

            def inner():
                return 10

            def innerinner(x):
                return x()
            return consumer_func(inner, innerinner)
        return impl
    cfunc = njit(impl_factory(consumer2arg))
    impl = impl_factory(consumer2arg.py_func)
    self.assertEqual(impl(), cfunc())