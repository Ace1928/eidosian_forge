import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_non_finite_inputs_and_int_bins(self):
    x = self.x
    u = self.u
    orig = u[0]
    u[0] = np.inf
    assert_raises(ValueError, binned_statistic, u, x, 'std', bins=10)
    assert_raises(ValueError, binned_statistic, u, x, 'std', bins=np.int64(10))
    u[0] = np.nan
    assert_raises(ValueError, binned_statistic, u, x, 'count', bins=10)
    u[0] = orig