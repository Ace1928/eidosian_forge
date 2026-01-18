import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_dd_sum(self):
    X = self.X
    v = self.v
    sum1, edges1, bc = binned_statistic_dd(X, v, 'sum', bins=3)
    sum2, edges2 = np.histogramdd(X, bins=3, weights=v)
    sum3, edges3, bc = binned_statistic_dd(X, v, np.sum, bins=3)
    assert_allclose(sum1, sum2)
    assert_allclose(edges1, edges2)
    assert_allclose(sum1, sum3)
    assert_allclose(edges1, edges3)