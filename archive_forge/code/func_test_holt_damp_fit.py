from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
@pytest.mark.smoke
def test_holt_damp_fit(self):
    fit1 = SimpleExpSmoothing(self.livestock2_livestock, initialization_method='estimated').fit()
    mod4 = Holt(self.livestock2_livestock, damped_trend=True, initialization_method='estimated')
    fit4 = mod4.fit(damping_trend=0.98, method='least_squares')
    mod5 = Holt(self.livestock2_livestock, exponential=True, damped_trend=True, initialization_method='estimated')
    fit5 = mod5.fit()
    assert_almost_equal(fit1.params['smoothing_level'], 1.0, 2)
    assert_almost_equal(fit1.params['smoothing_trend'], np.nan, 2)
    assert_almost_equal(fit1.params['damping_trend'], np.nan, 2)
    assert_almost_equal(fit1.params['initial_level'], 263.96, 1)
    assert_almost_equal(fit1.params['initial_trend'], np.nan, 2)
    assert_almost_equal(fit1.sse, 6761.35, 2)
    assert isinstance(fit1.summary().as_text(), str)
    assert_almost_equal(fit4.params['smoothing_level'], 0.98, 2)
    assert_almost_equal(fit4.params['smoothing_trend'], 0.0, 2)
    assert_almost_equal(fit4.params['damping_trend'], 0.98, 2)
    assert_almost_equal(fit4.params['initial_level'], 257.36, 2)
    assert_almost_equal(fit4.params['initial_trend'], 6.64, 2)
    assert_almost_equal(fit4.sse, 6036.56, 2)
    assert isinstance(fit4.summary().as_text(), str)
    assert_almost_equal(fit5.params['smoothing_level'], 0.97, 2)
    assert_almost_equal(fit5.params['smoothing_trend'], 0.0, 2)
    assert_almost_equal(fit5.params['damping_trend'], 0.98, 2)
    assert_almost_equal(fit5.params['initial_level'], 258.95, 1)
    assert_almost_equal(fit5.params['initial_trend'], 1.04, 2)
    assert_almost_equal(fit5.sse, 6082.0, 0)
    assert isinstance(fit5.summary().as_text(), str)