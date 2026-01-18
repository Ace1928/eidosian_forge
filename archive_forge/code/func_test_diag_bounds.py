from numpy.testing import (
from numpy import (
import numpy as np
import pytest
def test_diag_bounds(self):
    A = [[1, 2], [3, 4], [5, 6]]
    assert_equal(diag(A, k=2), [])
    assert_equal(diag(A, k=1), [2])
    assert_equal(diag(A, k=0), [1, 4])
    assert_equal(diag(A, k=-1), [3, 6])
    assert_equal(diag(A, k=-2), [5])
    assert_equal(diag(A, k=-3), [])