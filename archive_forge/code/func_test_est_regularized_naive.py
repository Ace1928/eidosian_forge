import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def test_est_regularized_naive():
    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    res = _est_regularized_naive(mod, 0, 2, fit_kwds={'alpha': 0.5})
    assert_equal(res.shape, beta.shape)
    mod = GLM(y, X, family=Binomial())
    res = _est_regularized_naive(mod, 0, 2, fit_kwds={'alpha': 0.5})
    assert_equal(res.shape, beta.shape)