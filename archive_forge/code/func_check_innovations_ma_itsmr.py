import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import (
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
def check_innovations_ma_itsmr(lake):
    ia, _ = innovations(lake, 10, demean=True)
    desired = [1.0816255264, 0.7781248438, 0.536716443, 0.3291559246, 0.316003985, 0.251375455, 0.2051536531, 0.1441070313, 0.343186834, 0.1827400798]
    assert_allclose(ia[10].ma_params, desired)
    u, v = arma_innovations(np.array(lake) - np.mean(lake), ma_params=ia[10].ma_params, sigma2=1)
    desired_sigma2 = 0.4523684344
    assert_allclose(np.sum(u ** 2 / v) / len(u), desired_sigma2)