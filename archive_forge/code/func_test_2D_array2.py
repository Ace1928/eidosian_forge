import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_2D_array2(self):
    a = np.array([1, 2])
    b = np.array([1, 2])
    res = dstack([a, b])
    desired = np.array([[[1, 1], [2, 2]]])
    assert_array_equal(res, desired)