from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_dynamic_factor_end_time_varying_exog_shocks(reset_randomstate):
    endog = np.arange(1, 23).reshape(11, 2)
    exog = np.arange(1, 21) ** 2
    mod = dynamic_factor.DynamicFactor(endog[:10], k_factors=1, factor_order=1, exog=exog[:10])
    mod.ssm.filter_univariate = True
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    mod2 = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1, exog=exog[:11])
    mod2.ssm.filter_univariate = True
    res2 = mod2.filter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    nsimulations = 10
    measurement_shocks = np.random.normal(size=(nsimulations, mod.k_endog))
    state_shocks = np.random.normal(size=(nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]
    actual = res.simulate(nsimulations, exog=exog[10:], anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    actual2 = res2.simulate(nsimulations, exog=exog[11:], anchor=-1, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1] + state_shocks[i - 1]
    desired = desired + measurement_shocks + exog[10:, None]
    assert_allclose(actual, desired)
    assert_allclose(actual2, desired)
    mod_actual = mod.simulate(res.params, nsimulations, exog=exog[10:], anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    assert_allclose(mod_actual, desired)