import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake, oshorts
from statsmodels.tsa.arima.estimators.gls import gls
@pytest.mark.todo('Low priority: test full GLS against another package')
@pytest.mark.smoke
def test_alternate_arma_estimators_valid():
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    _, res_yw = gls(endog, exog=exog, order=(1, 0, 0), arma_estimator='yule_walker', n_iter=1)
    assert_equal(res_yw.arma_estimator, 'yule_walker')
    _, res_b = gls(endog, exog=exog, order=(1, 0, 0), arma_estimator='burg', n_iter=1)
    assert_equal(res_b.arma_estimator, 'burg')
    _, res_i = gls(endog, exog=exog, order=(0, 0, 1), arma_estimator='innovations', n_iter=1)
    assert_equal(res_i.arma_estimator, 'innovations')
    _, res_hr = gls(endog, exog=exog, order=(1, 0, 1), arma_estimator='hannan_rissanen', n_iter=1)
    assert_equal(res_hr.arma_estimator, 'hannan_rissanen')
    _, res_ss = gls(endog, exog=exog, order=(1, 0, 1), arma_estimator='statespace', n_iter=1)
    assert_equal(res_ss.arma_estimator, 'statespace')
    _, res_imle = gls(endog, exog=exog, order=(1, 0, 1), n_iter=1)
    assert_equal(res_imle.arma_estimator, 'innovations_mle')