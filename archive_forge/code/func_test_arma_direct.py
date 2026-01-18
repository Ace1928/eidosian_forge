from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_arma_direct():
    np.random.seed(10239)
    nobs = 100
    eps = np.random.normal(size=nobs)
    exog = np.random.normal(size=nobs)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod.simulate([0.5, 1.0], nobs + 1, state_shocks=np.r_[eps, 0], initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * desired[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    actual = mod.simulate([0.5, 1.0], nobs + 1, state_shocks=np.r_[eps, 0], initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)
    mod = sarimax.SARIMAX([0], order=(1, 0, 1))
    actual = mod.simulate([0.5, 0.2, 1.0], nobs + 1, state_shocks=np.r_[eps, 0], initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * desired[i - 1] + 0.2 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)
    mod = sarimax.SARIMAX([0], order=(1, 0, 1), trend='c')
    actual = mod.simulate([1.3, 0.5, 0.2, 1.0], nobs + 1, state_shocks=np.r_[eps, 0], initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = trend + 0.5 * desired[i - 1] + 0.2 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)
    mod = sarimax.SARIMAX(np.zeros(nobs + 1), order=(1, 0, 1), trend='ct')
    actual = mod.simulate([1.3, 0.2, 0.5, 0.2, 1.0], nobs + 1, state_shocks=np.r_[eps, 0], initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3 + 0.2 * (i + 1)
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = trend + 0.5 * desired[i - 1] + 0.2 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)
    mod = sarimax.SARIMAX(np.zeros(nobs + 1), exog=np.r_[0, exog], order=(1, 0, 1), trend='ct')
    actual = mod.simulate([1.3, 0.2, -0.5, 0.5, 0.2, 1.0], nobs + 1, state_shocks=np.r_[eps, 0], initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3 + 0.2 * (i + 1)
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = trend + 0.5 * desired[i - 1] + 0.2 * eps[i - 1] + eps[i]
    desired = desired - 0.5 * exog
    assert_allclose(actual[1:], desired)