import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_logit_map_crossed_formula():
    data = gen_crossed_logit_pandas(10, 10, 1, 0.5)
    fml = 'y ~ fe'
    fml_vc = {'a': '0 + C(a)', 'b': '0 + C(b)'}
    glmm = BinomialBayesMixedGLM.from_formula(fml, fml_vc, data, vcp_p=0.5)
    rslt = glmm.fit_map()
    assert_allclose(glmm.logposterior_grad(rslt.params), np.zeros_like(rslt.params), atol=0.0001)
    rslt.summary()
    r = rslt.random_effects('a')
    assert_allclose(r.iloc[0, :].values, np.r_[-0.02004904, 0.094014], atol=0.0001)
    cm = rslt.cov_params()
    p = rslt.params.shape[0]
    assert_equal(list(cm.shape), [p, p])
    np.linalg.cholesky(cm)