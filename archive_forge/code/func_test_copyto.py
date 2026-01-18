import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
def test_copyto():
    a = np.arange(6, dtype='i4').reshape(2, 3)
    np.copyto(a, [[3, 1, 5], [6, 2, 1]])
    assert_equal(a, [[3, 1, 5], [6, 2, 1]])
    np.copyto(a[:, :2], a[::-1, 1::-1])
    assert_equal(a, [[2, 6, 5], [1, 3, 1]])
    assert_raises(TypeError, np.copyto, a, 1.5)
    np.copyto(a, 1.5, casting='unsafe')
    assert_equal(a, 1)
    np.copyto(a, 3, where=[True, False, True])
    assert_equal(a, [[3, 1, 3], [3, 1, 3]])
    assert_raises(TypeError, np.copyto, a, 3.5, where=[True, False, True])
    np.copyto(a, 4.0, casting='unsafe', where=[[0, 1, 1], [1, 0, 0]])
    assert_equal(a, [[3, 4, 4], [4, 1, 3]])
    np.copyto(a[:, :2], a[::-1, 1::-1], where=[[0, 1], [1, 1]])
    assert_equal(a, [[3, 4, 4], [4, 3, 3]])
    assert_raises(TypeError, np.copyto, [1, 2, 3], [2, 3, 4])