from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_arma_lfilter():
    np.random.seed(10239)
    nobs = 100
    eps = np.random.normal(size=nobs)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod.simulate([0.5, 1.0], nobs + 1, state_shocks=np.r_[eps, 0], initial_state=np.zeros(mod.k_states))
    desired = lfilter([1], [1, -0.5], eps)
    assert_allclose(actual[1:], desired)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    actual = mod.simulate([0.5, 1.0], nobs + 1, state_shocks=np.r_[eps, 0], initial_state=np.zeros(mod.k_states))
    desired = lfilter([1, 0.5], [1], eps)
    assert_allclose(actual[1:], desired)
    mod = sarimax.SARIMAX([0], order=(1, 0, 1))
    actual = mod.simulate([0.5, 0.2, 1.0], nobs + 1, state_shocks=np.r_[eps, 0], initial_state=np.zeros(mod.k_states))
    desired = lfilter([1, 0.2], [1, -0.5], eps)
    assert_allclose(actual[1:], desired)