import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_raises
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.durbin_levinson import durbin_levinson
def check_itsmr(lake):
    dl, _ = durbin_levinson(lake, 5)
    assert_allclose(dl[0].params, np.var(lake))
    assert_allclose(dl[1].ar_params, [0.8319112104])
    assert_allclose(dl[2].ar_params, [1.0538248798, -0.2667516276])
    desired = [1.0887037577, -0.4045435867, 0.1307541335]
    assert_allclose(dl[3].ar_params, desired)
    desired = [1.0842506581, -0.39076602696, 0.09367609911, 0.03405704644]
    assert_allclose(dl[4].ar_params, desired)
    desired = [1.08213598501, -0.39658257147, 0.11793957728, -0.03326633983, 0.06209208707]
    assert_allclose(dl[5].ar_params, desired)
    u, v = arma_innovations(np.array(lake) - np.mean(lake), ar_params=dl[5].ar_params, sigma2=1)
    desired_sigma2 = 0.4716322564
    assert_allclose(np.sum(u ** 2 / v) / len(u), desired_sigma2)