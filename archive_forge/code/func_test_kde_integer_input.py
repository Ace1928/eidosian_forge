from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_kde_integer_input():
    """Regression test for #1181."""
    x1 = np.arange(5)
    kde = stats.gaussian_kde(x1)
    y_expected = [0.13480721, 0.18222869, 0.19514935, 0.18222869, 0.13480721]
    assert_array_almost_equal(kde(x1), y_expected, decimal=6)