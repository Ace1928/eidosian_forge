import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_end_values(self):
    """Ensure that end values are exact."""
    a = np.pad(np.ones(10).reshape(2, 5), (223, 123), mode='linear_ramp')
    assert_equal(a[:, 0], 0.0)
    assert_equal(a[:, -1], 0.0)
    assert_equal(a[0, :], 0.0)
    assert_equal(a[-1, :], 0.0)