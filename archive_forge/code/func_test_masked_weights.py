import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_masked_weights(self):
    a = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0], [1, 0, 0], [0, 0, 0]])
    weights_unmasked = masked_array([5, 28, 31], mask=False)
    weights_masked = masked_array([5, 28, 31], mask=[1, 0, 0])
    avg_unmasked = average(a, axis=0, weights=weights_unmasked, returned=False)
    expected_unmasked = np.array([6.0, 5.21875, 6.21875])
    assert_almost_equal(avg_unmasked, expected_unmasked)
    avg_masked = average(a, axis=0, weights=weights_masked, returned=False)
    expected_masked = np.array([6.0, 5.576271186440678, 6.576271186440678])
    assert_almost_equal(avg_masked, expected_masked)
    a = np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[False, False, True, True])
    avg_unmasked = average(a, weights=[1, 1, 1, np.nan])
    assert_almost_equal(avg_unmasked, 1.5)
    a = np.ma.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 1.0, 2.0, 3.0]], mask=[[False, True, True, False], [True, False, True, True], [True, False, True, False]])
    avg_masked = np.ma.average(a, weights=[1, np.nan, 1], axis=0)
    avg_expected = np.ma.array([1.0, np.nan, np.nan, 3.5], mask=[False, True, True, False])
    assert_almost_equal(avg_masked, avg_expected)
    assert_equal(avg_masked.mask, avg_expected.mask)