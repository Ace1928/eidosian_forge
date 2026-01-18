import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_manyways(self):
    a = np.array([5, 7, 1, 2, 8])
    b = np.array([9, 8, 2, 4, 3, 1, 5])
    c1 = setxor1d(a, b)
    aux1 = intersect1d(a, b)
    aux2 = union1d(a, b)
    c2 = setdiff1d(aux2, aux1)
    assert_array_equal(c1, c2)