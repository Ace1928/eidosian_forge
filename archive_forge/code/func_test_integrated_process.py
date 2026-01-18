import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.innovations import _arma_innovations, arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX
@pytest.mark.parametrize('ar_params,diff,ma_params,sigma2', [(np.array([]), 1, np.array([]), 1), (np.array([0.0]), 1, np.array([0.0]), 1), (np.array([0.9]), 1, np.array([]), 1), (np.array([]), 1, np.array([0.9]), 1), (np.array([0.2, -0.4, 0.1, 0.1]), 1, np.array([0.5, 0.1]), 1.123), (np.array([0.5, 0.1]), 1, np.array([0.2, -0.4, 0.1, 0.1]), 1.123), (np.array([0.5, 0.1]), 2, np.array([0.2, -0.4, 0.1, 0.1]), 1.123)])
def test_integrated_process(ar_params, diff, ma_params, sigma2):
    nobs = 100
    endog = np.cumsum(np.random.normal(size=nobs))
    llf_obs = arma_innovations.arma_loglikeobs(np.diff(endog, diff), ar_params, ma_params, sigma2)
    mod = SARIMAX(endog, order=(len(ar_params), diff, len(ma_params)), simple_differencing=True)
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])
    assert_allclose(llf_obs, res.llf_obs)