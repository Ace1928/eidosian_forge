import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_dict_use_with_none_value(self):

    @njit
    def foo():
        k = {1: None}
        return k
    with self.assertRaises(TypingError) as raises:
        foo()
    self.assertIn('Dict.value_type cannot be of type none', str(raises.exception))