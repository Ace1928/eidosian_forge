import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_raises
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.durbin_levinson import durbin_levinson
@pytest.mark.low_precision('Test against Example 5.1.1 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_511():
    endog = dowj.diff().iloc[1:]
    dl, _ = durbin_levinson(endog, ar_order=2, demean=True)
    assert_allclose(dl[0].params, np.var(endog))
    assert_allclose(dl[1].params, [0.4219, 0.1479], atol=0.0001)
    assert_allclose(dl[2].params, [0.3739, 0.1138, 0.146], atol=0.0001)