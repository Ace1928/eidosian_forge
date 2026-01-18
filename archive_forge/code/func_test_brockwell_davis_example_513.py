import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.burg import burg
@pytest.mark.low_precision('Test against Example 5.1.3 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_513():
    endog = dowj.diff().iloc[1:]
    res, _ = burg(endog, ar_order=1, demean=True)
    assert_allclose(res.ar_params, [0.4371], atol=0.0001)
    assert_allclose(res.sigma2, 0.1423, atol=0.0001)