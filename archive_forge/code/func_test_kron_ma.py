import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_kron_ma(self):
    x = np.ma.array([[1, 2], [3, 4]], mask=[[0, 1], [1, 0]])
    k = np.ma.array(np.diag([1, 4, 4, 16]), mask=~np.array(np.identity(4), dtype=bool))
    assert_array_equal(k, np.kron(x, x))