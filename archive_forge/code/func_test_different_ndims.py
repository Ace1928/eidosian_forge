import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_different_ndims(self, block):
    a = 1.0
    b = 2 * np.ones((1, 2))
    c = 3 * np.ones((1, 1, 3))
    result = block([a, b, c])
    expected = np.array([[[1.0, 2.0, 2.0, 3.0, 3.0, 3.0]]])
    assert_equal(result, expected)