from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def check_standardized_results(res1, res2, check_diagnostics=True):
    mod1 = res1.model
    mod2 = res2.model
    tmp = (1 - res1.filter_results.missing.T) * np.array(mod1._endog_std)[None, :] ** 2
    mask = res1.filter_results.missing.T.astype(bool)
    tmp[mask] = 1.0
    llf_obs_diff = -0.5 * np.log(tmp).sum(axis=1)
    assert_allclose(res1.llf_obs + llf_obs_diff, res2.llf_obs)
    assert_allclose(res1.mae, res2.mae)
    assert_allclose(res1.mse, res2.mse)
    assert_allclose(res1.sse, res2.sse)
    std = np.array(mod1._endog_std)
    mean = np.array(mod1._endog_mean)
    if mod1.k_endog > 1:
        std = std[None, :]
        mean = mean[None, :]
    if mod1.k_endog == 1:
        assert_allclose(res1.fittedvalues.shape, (mod1.nobs,))
    else:
        assert_allclose(res1.fittedvalues.shape, (mod1.nobs, mod1.k_endog))
    actual = np.array(res1.fittedvalues) * std + mean
    assert_allclose(actual, res2.fittedvalues)
    actual = np.array(res1.resid) * std
    assert_allclose(actual, res2.resid)
    if check_diagnostics:
        actual = res1.test_normality(method='jarquebera')
        desired = res2.test_normality(method='jarquebera')
        assert_allclose(actual, desired)
        actual = res1.test_heteroskedasticity(method='breakvar')
        desired = res2.test_heteroskedasticity(method='breakvar')
        assert_allclose(actual, desired)
        lags = min(10, res1.nobs_effective // 5)
        actual = res1.test_serial_correlation(method='ljungbox', lags=lags)
        desired = res2.test_serial_correlation(method='ljungbox', lags=lags)
        assert_allclose(actual, desired)
    start = res1.nobs // 10
    dynamic = res1.nobs // 10
    end = res1.nobs + 10
    predict_actual = res1.predict()
    forecast_actual = res1.forecast(10)
    predict_dynamic_forecast_actual = res1.predict(start=start, end=end, dynamic=dynamic)
    get_predict_actual = res1.get_prediction()
    get_forecast_actual = res1.get_forecast(10)
    get_predict_dynamic_forecast_actual = res1.get_prediction(start=start, end=end, dynamic=dynamic)
    predict_desired = res2.predict()
    forecast_desired = res2.forecast(10)
    predict_dynamic_forecast_desired = res2.predict(start=start, end=end, dynamic=dynamic)
    get_predict_desired = res2.get_prediction()
    get_forecast_desired = res2.get_forecast(10)
    get_predict_dynamic_forecast_desired = res2.get_prediction(start=start, end=end, dynamic=dynamic)
    assert_allclose(predict_actual, predict_desired)
    assert_allclose(forecast_actual, forecast_desired)
    assert_allclose(predict_dynamic_forecast_actual, predict_dynamic_forecast_desired)
    for i in range(mod1.k_endog):
        assert_allclose(get_predict_actual.summary_frame(endog=i), get_predict_desired.summary_frame(endog=i))
        assert_allclose(get_forecast_actual.summary_frame(endog=i), get_forecast_desired.summary_frame(endog=i))
        assert_allclose(get_predict_dynamic_forecast_actual.summary_frame(endog=i), get_predict_dynamic_forecast_desired.summary_frame(endog=i))
    np.random.seed(1234)
    nsimulations = 100
    initial_state = np.random.multivariate_normal(res1.filter_results.initial_state, res1.filter_results.initial_state_cov)
    raw_measurement_shocks = np.random.multivariate_normal(np.zeros(mod1.k_endog), np.eye(mod1.k_endog), size=nsimulations)
    state_shocks = np.random.multivariate_normal(np.zeros(mod1.ssm.k_posdef), mod1['state_cov'], size=nsimulations)
    L1 = np.diag(mod1['obs_cov'].diagonal() ** 0.5)
    measurement_shocks1 = (L1 @ raw_measurement_shocks.T).T
    L2 = np.diag(mod2['obs_cov'].diagonal() ** 0.5)
    measurement_shocks2 = (L2 @ raw_measurement_shocks.T).T
    sim_actual = res1.simulate(nsimulations=nsimulations, initial_state=initial_state, measurement_shocks=measurement_shocks1, state_shocks=state_shocks)
    sim_desired = res2.simulate(nsimulations=nsimulations, initial_state=initial_state, measurement_shocks=measurement_shocks2, state_shocks=state_shocks)
    assert_allclose(sim_actual, sim_desired)
    sim_actual = res1.simulate(nsimulations=nsimulations, initial_state=initial_state, measurement_shocks=measurement_shocks1, state_shocks=state_shocks, anchor='end')
    sim_desired = res2.simulate(nsimulations=nsimulations, initial_state=initial_state, measurement_shocks=measurement_shocks2, state_shocks=state_shocks, anchor='end')
    assert_allclose(sim_actual, sim_desired)
    irfs_actual = res1.impulse_responses(10)
    irfs_desired = res2.impulse_responses(10)
    assert_allclose(irfs_actual, irfs_desired)
    irfs_actual = res1.impulse_responses(10, orthogonalized=True)
    irfs_desired = res2.impulse_responses(10, orthogonalized=True)
    assert_allclose(irfs_actual, irfs_desired)
    irfs_actual = res1.impulse_responses(10, cumulative=True)
    irfs_desired = res2.impulse_responses(10, cumulative=True)
    assert_allclose(irfs_actual, irfs_desired)
    irfs_actual = res1.impulse_responses(10, orthogonalized=True, cumulative=True)
    irfs_desired = res2.impulse_responses(10, orthogonalized=True, cumulative=True)
    assert_allclose(irfs_actual, irfs_desired)
    irfs_actual = res1.impulse_responses(10, orthogonalized=True, cumulative=True, anchor='end')
    irfs_desired = res2.impulse_responses(10, orthogonalized=True, cumulative=True, anchor='end')
    assert_allclose(irfs_actual, irfs_desired)