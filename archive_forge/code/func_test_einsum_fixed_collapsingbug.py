import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_fixed_collapsingbug(self):
    x = np.random.normal(0, 1, (5, 5, 5, 5))
    y1 = np.zeros((5, 5))
    np.einsum('aabb->ab', x, out=y1)
    idx = np.arange(5)
    y2 = x[idx[:, None], idx[:, None], idx, idx]
    assert_equal(y1, y2)