import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_binned_statistic_float32(self):
    X = np.array([0, 0.42358226], dtype=np.float32)
    stat, _, _ = binned_statistic(X, None, 'count', bins=5)
    assert_allclose(stat, np.array([1, 0, 0, 0, 1], dtype=np.float64))