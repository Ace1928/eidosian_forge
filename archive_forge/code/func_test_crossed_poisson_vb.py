import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_crossed_poisson_vb():
    y, exog_fe, exog_vc, ident = gen_crossed_poisson(10, 10, 1, 0.5)
    glmm1 = PoissonBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5, fe_p=0.5)
    rslt1 = glmm1.fit_map()
    glmm2 = PoissonBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5, fe_p=0.5)
    rslt2 = glmm2.fit_vb(mean=rslt1.params)
    rslt1.summary()
    rslt2.summary()
    assert_allclose(rslt1.params[0:5], np.r_[-0.54855281, 0.10458834, -0.68777741, -0.01699925, 0.77200546], rtol=0.0001, atol=0.0001)
    assert_allclose(rslt2.params[0:5], np.r_[-0.54691502, 0.22297158, -0.52673802, -0.06218684, 0.74385237], rtol=0.0001, atol=0.0001)
    for rslt in (rslt1, rslt2):
        cp = rslt.cov_params()
        p = len(rslt.params)
        if rslt is rslt1:
            assert_equal(cp.shape, np.r_[p, p])
            np.linalg.cholesky(cp)
        else:
            assert_equal(cp.shape, np.r_[p,])
            assert_equal(cp > 0, True * np.ones(p))