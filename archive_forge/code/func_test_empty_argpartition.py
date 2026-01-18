import sys
import pytest
import numpy as np
from numpy.testing import (
def test_empty_argpartition(self):
    a = np.array([0, 2, 4, 6, 8, 10])
    a = a.argpartition(np.array([], dtype=np.int16))
    b = np.array([0, 1, 2, 3, 4, 5])
    assert_array_equal(a, b)