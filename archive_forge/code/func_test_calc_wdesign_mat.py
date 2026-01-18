import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def test_calc_wdesign_mat():
    np.random.seed(435265)
    X = np.random.normal(size=(3, 3))
    y = np.random.randint(0, 2, size=3)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    dmat = _calc_wdesign_mat(mod, beta, {})
    assert_allclose(dmat, np.array([[1.306314, -0.024897, 1.326498], [-0.539219, -0.483028, -0.703503], [-3.327987, 0.524541, -0.139761]]), atol=1e-06, rtol=0)
    mod = GLM(y, X, family=Binomial())
    dmat = _calc_wdesign_mat(mod, beta, {})
    assert_allclose(dmat, np.array([[0.408616, -0.007788, 0.41493], [-0.263292, -0.235854, -0.343509], [-0.11241, 0.017718, -0.004721]]), atol=1e-06, rtol=0)