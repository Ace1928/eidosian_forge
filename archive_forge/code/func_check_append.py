from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def check_append(res1, res2, endog_M2, endog_Q2):
    res1_append = res1.append(endog_M2, endog_quarterly=endog_Q2)
    mod1_append = res1_append.model
    mod1 = res1.model
    check_identical_models(mod1, mod1_append, check_nobs=False)
    assert_equal(mod1_append.nobs, mod1.nobs + len(endog_M2))
    assert_allclose(mod1_append.endog[:mod1.nobs], mod1.endog)
    assert_allclose(res1_append.filter_results.initial_state_cov, res1.filter_results.initial_state_cov)
    assert_allclose(res1_append.llf_obs[:mod1.nobs], res1.llf_obs)
    assert_allclose(res1_append.filter_results.forecasts[:, :mod1.nobs], res1.filter_results.forecasts)
    assert_allclose(res1_append.filter_results.forecasts_error[:, :mod1.nobs], res1.filter_results.forecasts_error)
    assert_allclose(res1_append.filter_results.initial_state, res1.filter_results.initial_state)
    assert_allclose(res1_append.filter_results.initial_state_cov, res1.filter_results.initial_state_cov)
    assert_allclose(res1_append.filter_results.filtered_state[:, :mod1.nobs], res1.filter_results.filtered_state)
    assert_allclose(res1_append.filter_results.filtered_state_cov[..., :mod1.nobs], res1.filter_results.filtered_state_cov)
    res2_append = res2.append(endog_M2, endog_quarterly=endog_Q2)
    mod2_append = res2_append.model
    mod2 = res2.model
    mod2_append.update(res2_append.params)
    mod2_append['obs_intercept'] = mod2['obs_intercept']
    mod2_append['design'] = mod2['design']
    mod2_append['obs_cov'] = mod2['obs_cov']
    mod2_append.update = lambda params, **kwargs: params
    res2_append = mod2_append.smooth(res2_append.params)
    check_identical_models(mod2, mod2_append, check_nobs=False)
    assert_allclose(res2_append.filter_results.initial_state_cov, res2.filter_results.initial_state_cov)
    assert_allclose(res2_append.llf_obs[:mod2.nobs], res2.llf_obs)
    assert_allclose(res2_append.filter_results.forecasts[:, :mod2.nobs], res2.filter_results.forecasts)
    assert_allclose(res2_append.filter_results.forecasts_error[:, :mod2.nobs], res2.filter_results.forecasts_error)
    assert_allclose(res2_append.filter_results.initial_state, res2.filter_results.initial_state)
    assert_allclose(res2_append.filter_results.initial_state_cov, res2.filter_results.initial_state_cov)
    assert_allclose(res2_append.filter_results.filtered_state[:, :mod2.nobs], res2.filter_results.filtered_state)
    assert_allclose(res2_append.filter_results.filtered_state_cov[..., :mod2.nobs], res2.filter_results.filtered_state_cov)
    check_standardized_results(res1_append, res2_append)