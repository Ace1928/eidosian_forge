import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_check_03(self):
    a = np.pad([1, 2, 3], 6, 'symmetric')
    b = np.array([1, 2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3])
    assert_array_equal(a, b)