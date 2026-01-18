import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_crossed_logit_vb_formula():
    data = gen_crossed_logit_pandas(10, 10, 1, 2)
    fml = 'y ~ fe'
    fml_vc = {'a': '0 + C(a)', 'b': '0 + C(b)'}
    glmm1 = BinomialBayesMixedGLM.from_formula(fml, fml_vc, data, vcp_p=0.5)
    rslt1 = glmm1.fit_vb()
    glmm2 = BinomialBayesMixedGLM(glmm1.endog, glmm1.exog, glmm1.exog_vc, glmm1.ident, vcp_p=0.5)
    rslt2 = glmm2.fit_vb()
    assert_allclose(rslt1.params, rslt2.params, atol=0.0001)
    rslt1.summary()
    rslt2.summary()
    for rslt in (rslt1, rslt2):
        cp = rslt.cov_params()
        p = len(rslt.params)
        if rslt is rslt1:
            assert_equal(cp.shape, np.r_[p,])
            assert_equal(cp > 0, True * np.ones(p))
        else:
            assert_equal(cp.shape, np.r_[p,])
            assert_equal(cp > 0, True * np.ones(p))