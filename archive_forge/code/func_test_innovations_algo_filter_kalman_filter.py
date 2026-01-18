import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.innovations import _arma_innovations, arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX
@pytest.mark.parametrize('ar_params,ma_params,sigma2', [(np.array([]), np.array([]), 1), (np.array([0.0]), np.array([0.0]), 1), (np.array([0.9]), np.array([]), 1), (np.array([]), np.array([0.9]), 1), (np.array([0.2, -0.4, 0.1, 0.1]), np.array([0.5, 0.1]), 1.123), (np.array([0.5, 0.1]), np.array([0.2, -0.4, 0.1, 0.1]), 1.123)])
def test_innovations_algo_filter_kalman_filter(ar_params, ma_params, sigma2):
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]
    endog = np.random.normal(size=10)
    nobs = len(endog)
    arma_process_acovf = arma_acovf(ar, ma, nobs=nobs, sigma2=sigma2)
    transformed_acov = _arma_innovations.darma_transformed_acovf_fast(ar, ma, arma_process_acovf / sigma2)
    acovf, acovf2 = (np.array(mv) for mv in transformed_acov)
    theta, r = _arma_innovations.darma_innovations_algo_fast(nobs, ar_params, ma_params, acovf, acovf2)
    u = _arma_innovations.darma_innovations_filter(endog, ar_params, ma_params, theta)
    v = np.array(r) * sigma2
    u = np.array(u)
    llf_obs = -0.5 * u ** 2 / v - 0.5 * np.log(2 * np.pi * v)
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])
    assert_allclose(u, res.forecasts_error[0])
    assert_allclose(llf_obs, res.llf_obs)
    llf_obs2 = _arma_innovations.darma_loglikeobs_fast(endog, ar_params, ma_params, sigma2)
    assert_allclose(llf_obs2, res.llf_obs)