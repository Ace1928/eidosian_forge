import numpy as np
from statsmodels.discrete.conditional_models import (
from statsmodels.tools.numdiff import approx_fprime
from numpy.testing import assert_allclose
import pandas as pd
def test_logit_2d():
    y = np.r_[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    x1 = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x2 = np.r_[0, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    x = np.empty((10, 2))
    x[:, 0] = x1
    x[:, 1] = x2
    model = ConditionalLogit(y, x, groups=g)
    for x in (-1, 0, 1, 2):
        params = np.r_[x, -1.5 * x]
        _, grad = model._denom_grad(0, params)
        ngrad = approx_fprime(params, lambda x: model._denom(0, x))
        assert_allclose(grad, ngrad, rtol=1e-05)
    for x in (-1, 0, 1, 2):
        params = np.r_[-0.5 * x, 0.5 * x]
        grad = approx_fprime(params, model.loglike)
        score = model.score(params)
        assert_allclose(grad, score, rtol=0.0001)
    result = model.fit()
    assert_allclose(result.params, np.r_[1.011074, 1.236758], rtol=0.001)
    assert_allclose(result.bse, np.r_[1.420784, 1.361738], rtol=1e-05)
    result.summary()