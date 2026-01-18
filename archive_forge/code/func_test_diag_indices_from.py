import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_diag_indices_from(self):
    x = np.random.random((4, 4))
    r, c = diag_indices_from(x)
    assert_array_equal(r, np.arange(4))
    assert_array_equal(c, np.arange(4))