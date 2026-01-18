import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_2d_max(self):
    x = self.x
    y = self.y
    v = self.v
    stat1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'max', bins=5)
    stat2, binx2, biny2, bc = binned_statistic_2d(x, y, v, np.max, bins=5)
    assert_allclose(stat1, stat2)
    assert_allclose(binx1, binx2)
    assert_allclose(biny1, biny2)