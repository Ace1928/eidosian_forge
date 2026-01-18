import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.special import xlogy
from scipy.stats.contingency import (margins, expected_freq,
def test_expected_freq():
    assert_array_equal(expected_freq([1]), np.array([1.0]))
    observed = np.array([[[2, 0], [0, 2]], [[0, 2], [2, 0]], [[1, 1], [1, 1]]])
    e = expected_freq(observed)
    assert_array_equal(e, np.ones_like(observed))
    observed = np.array([[10, 10, 20], [20, 20, 20]])
    e = expected_freq(observed)
    correct = np.array([[12.0, 12.0, 16.0], [18.0, 18.0, 24.0]])
    assert_array_almost_equal(e, correct)