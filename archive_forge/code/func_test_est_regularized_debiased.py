import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def test_est_regularized_debiased():
    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    res = _est_regularized_debiased(mod, 0, 2, fit_kwds={'alpha': 0.5})
    bhat = res[0]
    grad = res[1]
    ghat_l = res[2]
    that_l = res[3]
    assert_(isinstance(res, tuple))
    assert_equal(bhat.shape, beta.shape)
    assert_equal(grad.shape, beta.shape)
    assert_(isinstance(ghat_l, list))
    assert_(isinstance(that_l, list))
    assert_equal(len(ghat_l), len(that_l))
    assert_equal(ghat_l[0].shape, (2,))
    assert_(isinstance(that_l[0], float))
    mod = GLM(y, X, family=Binomial())
    res = _est_regularized_debiased(mod, 0, 2, fit_kwds={'alpha': 0.5})
    bhat = res[0]
    grad = res[1]
    ghat_l = res[2]
    that_l = res[3]
    assert_(isinstance(res, tuple))
    assert_equal(bhat.shape, beta.shape)
    assert_equal(grad.shape, beta.shape)
    assert_(isinstance(ghat_l, list))
    assert_(isinstance(that_l, list))
    assert_equal(len(ghat_l), len(that_l))
    assert_equal(ghat_l[0].shape, (2,))
    assert_(isinstance(that_l[0], float))