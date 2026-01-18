import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_2d_binnumbers_unraveled(self):
    x = self.x
    y = self.y
    v = self.v
    stat, edgesx, bcx = binned_statistic(x, v, 'mean', bins=20)
    stat, edgesy, bcy = binned_statistic(y, v, 'mean', bins=10)
    stat2, edgesx2, edgesy2, bc2 = binned_statistic_2d(x, y, v, 'mean', bins=(20, 10), expand_binnumbers=True)
    bcx3 = np.searchsorted(edgesx, x, side='right')
    bcy3 = np.searchsorted(edgesy, y, side='right')
    bcx3[x == x.max()] -= 1
    bcy3[y == y.max()] -= 1
    assert_allclose(bcx, bc2[0])
    assert_allclose(bcy, bc2[1])
    assert_allclose(bcx3, bc2[0])
    assert_allclose(bcy3, bc2[1])