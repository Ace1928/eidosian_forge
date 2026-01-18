import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_array_like_values(self):
    a = np.zeros((5, 5))
    s = np.arange(25, dtype=np.float64).reshape(5, 5)
    a[[0, 1, 2, 3, 4], :] = memoryview(s)
    assert_array_equal(a, s)
    a[:, [0, 1, 2, 3, 4]] = memoryview(s)
    assert_array_equal(a, s)
    a[...] = memoryview(s)
    assert_array_equal(a, s)