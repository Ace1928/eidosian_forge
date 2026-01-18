from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
def test_multiply_defined_freevar(self):

    @njit
    def impl(c):
        if c:
            x = 3

            def inner(y):
                return y + x
            r = consumer(inner, 1)
        else:
            x = 6

            def inner(y):
                return y + x
            r = consumer(inner, 2)
        return r
    with self.assertRaises(errors.TypingError) as e:
        impl(1)
    self.assertIn('Cannot capture a constant value for variable', str(e.exception))