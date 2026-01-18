from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
def test_escape_with_kwargs_override_kwargs(self):

    @njit
    def specialised_consumer(func, *args):
        x, y, z = args
        a = func(x, y, z, mydefault1=1000)
        b = func(x, y, z, mydefault2=1000)
        c = func(x, y, z, mydefault1=1000, mydefault2=1000)
        return a + b + c

    def impl_factory(consumer_func):

        def impl():
            t = 12

            def inner(a, b, c, mydefault1=123, mydefault2=456):
                z = 4
                return mydefault1 + mydefault2 + z + t + a + b + c
            return (inner(1, 2, 5, 91, 53), consumer_func(inner, 1, 2, 11), consumer_func(inner, 1, 2, 3), inner(1, 2, 4))
        return impl
    cfunc = njit(impl_factory(specialised_consumer))
    impl = impl_factory(specialised_consumer.py_func)
    np.testing.assert_allclose(impl(), cfunc())