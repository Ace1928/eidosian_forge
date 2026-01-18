import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_1d_bincode(self):
    x = self.x[:20]
    v = self.v[:20]
    count1, edges1, bc = binned_statistic(x, v, 'count', bins=3)
    bc2 = np.array([3, 2, 1, 3, 2, 3, 3, 3, 3, 1, 1, 3, 3, 1, 2, 3, 1, 1, 2, 1])
    bcount = [(bc == i).sum() for i in np.unique(bc)]
    assert_allclose(bc, bc2)
    assert_allclose(bcount, count1)