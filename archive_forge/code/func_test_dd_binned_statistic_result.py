import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_dd_binned_statistic_result(self):
    x = np.random.random((10000, 3))
    v = np.random.random(10000)
    bins = np.linspace(0, 1, 10)
    bins = (bins, bins, bins)
    result = binned_statistic_dd(x, v, 'mean', bins=bins)
    stat = result.statistic
    result = binned_statistic_dd(x, v, 'mean', binned_statistic_result=result)
    stat2 = result.statistic
    assert_allclose(stat, stat2)