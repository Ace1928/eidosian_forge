import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_2d_multi_values(self):
    x = self.x
    y = self.y
    v = self.v
    w = self.w
    stat1v, binx1v, biny1v, bc1v = binned_statistic_2d(x, y, v, 'mean', bins=8)
    stat1w, binx1w, biny1w, bc1w = binned_statistic_2d(x, y, w, 'mean', bins=8)
    stat2, binx2, biny2, bc2 = binned_statistic_2d(x, y, [v, w], 'mean', bins=8)
    assert_allclose(stat2[0], stat1v)
    assert_allclose(stat2[1], stat1w)
    assert_allclose(binx1v, binx2)
    assert_allclose(biny1w, biny2)
    assert_allclose(bc1v, bc2)