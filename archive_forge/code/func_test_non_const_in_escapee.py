from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
def test_non_const_in_escapee(self):

    @njit
    def impl(x):
        z = np.arange(x)

        def inner(val):
            return 1 + z + val
        return consumer(inner, x)
    with self.assertRaises(errors.TypingError) as e:
        impl(1)
    self.assertIn('Cannot capture the non-constant value associated', str(e.exception))