import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import (
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
def test_innovations_mle_statespace_seasonal():
    endog = lake.copy()
    endog = endog - endog.mean()
    start_params = [0, np.var(endog)]
    p, mleres = innovations_mle(endog, seasonal_order=(1, 0, 0, 4), demean=False, start_params=start_params)
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), seasonal_order=(1, 0, 0, 4))
    res = mod.filter(p.params)
    assert_allclose(-mleres.minimize_results.fun, res.llf)
    res2 = mod.fit(start_params=p.params, disp=0)
    assert_allclose(p.params, res2.params)
    p2, _ = innovations_mle(endog, seasonal_order=(1, 0, 0, 4), demean=False)
    assert_allclose(p.params, p2.params, atol=1e-05)