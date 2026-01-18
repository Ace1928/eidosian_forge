import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_c_():
    a = np.c_[np.array([[1, 2, 3]]), 0, 0, np.array([[4, 5, 6]])]
    assert_equal(a, [[1, 2, 3, 0, 0, 4, 5, 6]])