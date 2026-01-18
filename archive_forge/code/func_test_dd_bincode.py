import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_dd_bincode(self):
    X = self.X[:20]
    v = self.v[:20]
    count1, edges1, bc = binned_statistic_dd(X, v, 'count', bins=3)
    bc2 = np.array([63, 33, 86, 83, 88, 67, 57, 33, 42, 41, 82, 83, 92, 32, 36, 91, 43, 87, 81, 81])
    bcount = [(bc == i).sum() for i in np.unique(bc)]
    assert_allclose(bc, bc2)
    count1adj = count1[count1.nonzero()]
    assert_allclose(bcount, count1adj)