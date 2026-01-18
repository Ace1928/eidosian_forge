import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_getitem2(self):
    cgetitem2 = jit(nopython=True)(getitem2)
    arr = np.array(b'12')
    self.assertPreciseEqual(cgetitem2(arr, (), 0), getitem2(arr, (), 0))
    with self.assertRaisesRegex(IndexError, 'index out of range'):
        cgetitem2(arr, (), 2)
    arr = np.array('12')
    self.assertPreciseEqual(cgetitem2(arr, (), 0), getitem2(arr, (), 0))
    with self.assertRaisesRegex(IndexError, 'index out of range'):
        cgetitem2(arr, (), 2)
    arr = np.array([b'12', b'3'])
    self.assertPreciseEqual(cgetitem2(arr, 0, 0), getitem2(arr, 0, 0))
    self.assertPreciseEqual(cgetitem2(arr, 0, 1), getitem2(arr, 0, 1))
    self.assertPreciseEqual(cgetitem2(arr, 1, 0), getitem2(arr, 1, 0))
    with self.assertRaisesRegex(IndexError, 'index out of range'):
        cgetitem2(arr, 1, 1)
    arr = np.array(['12', '3'])
    self.assertPreciseEqual(cgetitem2(arr, 0, 0), getitem2(arr, 0, 0))
    self.assertPreciseEqual(cgetitem2(arr, 0, 1), getitem2(arr, 0, 1))
    self.assertPreciseEqual(cgetitem2(arr, 1, 0), getitem2(arr, 1, 0))
    with self.assertRaisesRegex(IndexError, 'index out of range'):
        cgetitem2(arr, 1, 1)