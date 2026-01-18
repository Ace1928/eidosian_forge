import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_n_zero_stream(self):
    np.random.seed(8675309)
    expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 4, 2, 3, 3, 1, 5, 3, 1, 3]])
    assert_array_equal(random.binomial([[0], [10]], 0.25, size=(2, 10)), expected)