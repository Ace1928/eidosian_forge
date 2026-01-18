import numpy as np
from statsmodels.discrete.conditional_models import (
from statsmodels.tools.numdiff import approx_fprime
from numpy.testing import assert_allclose
import pandas as pd
def test_logit_1d():
    y = np.r_[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    x = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x = x[:, None]
    model = ConditionalLogit(y, x, groups=g)
    for x in (-1, 0, 1, 2):
        params = np.r_[x,]
        _, grad = model._denom_grad(0, params)
        ngrad = approx_fprime(params, lambda x: model._denom(0, x)).squeeze()
        assert_allclose(grad, ngrad)
    for x in (-1, 0, 1, 2):
        grad = approx_fprime(np.r_[x,], model.loglike).squeeze()
        score = model.score(np.r_[x,])
        assert_allclose(grad, score, rtol=0.0001)
    result = model.fit()
    assert_allclose(result.params, np.r_[0.9272407], rtol=1e-05)
    assert_allclose(result.bse, np.r_[1.295155], rtol=1e-05)