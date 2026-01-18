import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_, assert_equal
from statsmodels.stats import moment_helpers
from statsmodels.stats.moment_helpers import (cov2corr, mvsk2mc, mc2mvsk,
@pytest.mark.parametrize('test_vals', multidimension_test_vals)
def test_multidimensional(test_vals):
    assert_almost_equal(cum2mc(mnc2cum(mc2mnc(test_vals).T).T).T, test_vals)
    assert_almost_equal(cum2mc(mc2cum(test_vals).T).T, test_vals)
    assert_almost_equal(mvsk2mc(mc2mvsk(test_vals).T).T, test_vals)