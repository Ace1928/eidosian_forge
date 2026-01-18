import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose, assert_equal
from statsmodels.regression.linear_model import GLSAR
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata
def test_glsar_iter0(self):
    endog = self.res.model.endog
    exog = self.res.model.exog
    rho = np.array([0.207, 0.275, 1.033])
    mod1 = GLSAR(endog, exog, rho)
    res1 = mod1.fit()
    res0 = mod1.iterative_fit(0)
    res0b = mod1.iterative_fit(1)
    assert_allclose(res0.params, res1.params, rtol=1e-11)
    assert_allclose(res0b.params, res1.params, rtol=1e-11)
    assert_allclose(res0.model.rho, rho, rtol=1e-11)
    assert_allclose(res0b.model.rho, rho, rtol=1e-11)