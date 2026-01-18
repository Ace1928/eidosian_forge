import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX
@pytest.mark.parametrize('ar_params', ([1.9, -0.8], [1.0], [2.0, -1.0]))
def test_innovations_nonstationary(ar_params):
    np.random.seed(42)
    endog = np.random.normal(size=100)
    with pytest.raises(ValueError, match="The model's autoregressive"):
        arma_innovations.arma_innovations(endog, ar_params=ar_params)