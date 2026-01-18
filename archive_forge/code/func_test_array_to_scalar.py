import numpy as np
from numba.core.errors import TypingError
from numba import njit
from numba.core import types
import struct
import unittest
def test_array_to_scalar(self):
    """
        Ensure that a TypingError exception is raised if
        user tries to convert numpy array to scalar
        """
    with self.assertRaises(TypingError) as raises:
        njit(())(numpy_scalar_cast_error)
    self.assertIn('Casting array(float64, 1d, C) to int32 directly is unsupported.', str(raises.exception))