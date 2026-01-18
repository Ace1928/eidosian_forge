import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_2D_arrays(self):
    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    expected = np.array([[1, 2], [2, 3], [3, 4]])
    actual = np.column_stack((a, b))
    assert_equal(actual, expected)