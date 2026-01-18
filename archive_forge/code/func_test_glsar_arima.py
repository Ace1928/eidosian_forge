import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose, assert_equal
from statsmodels.regression.linear_model import GLSAR
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata
def test_glsar_arima(self):
    from statsmodels.tsa.arima.model import ARIMA
    endog = self.res.model.endog
    exog = self.res.model.exog
    mod1 = GLSAR(endog, exog, 3)
    res = mod1.iterative_fit(10)
    mod_arma = ARIMA(endog, order=(3, 0, 0), exog=exog[:, :-1])
    res_arma = mod_arma.fit()
    assert_allclose(res.params, res_arma.params[[1, 2, 0]], atol=0.01, rtol=0.01)
    assert_allclose(res.model.rho, res_arma.params[3:6], atol=0.05, rtol=0.001)
    assert_allclose(res.bse, res_arma.bse[[1, 2, 0]], atol=0.1, rtol=0.001)
    assert_equal(len(res.history['params']), 5)
    assert_equal(res.history['params'][-1], res.params)
    res2 = mod1.iterative_fit(4, rtol=0)
    assert_equal(len(res2.history['params']), 4)
    assert_equal(len(res2.history['rho']), 4)