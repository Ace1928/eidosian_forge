import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.dimred import (
from numpy.testing import (assert_equal, assert_allclose)
from statsmodels.tools.numdiff import approx_fprime
def test_covreduce():
    np.random.seed(34324)
    p = 4
    endog = []
    exog = []
    for k in range(3):
        c = np.eye(p)
        x = np.random.normal(size=(2, 2))
        c[0:2, 0:2] = np.dot(x.T, x)
        cr = np.linalg.cholesky(c)
        m = 1000 * k + 50 * k
        x = np.random.normal(size=(m, p))
        x = np.dot(x, cr.T)
        exog.append(x)
        endog.append(k * np.ones(m))
    endog = np.concatenate(endog)
    exog = np.concatenate(exog, axis=0)
    for dim in (1, 2, 3):
        cr = CORE(endog, exog, dim)
        pt = np.random.normal(size=(p, dim))
        pt, _, _ = np.linalg.svd(pt, 0)
        gn = approx_fprime(pt.ravel(), cr.loglike, 1e-07)
        g = cr.score(pt.ravel())
        assert_allclose(g, gn, 1e-05, 1e-05)
        rslt = cr.fit()
        proj = rslt.params
        assert_equal(proj.shape[0], p)
        assert_equal(proj.shape[1], dim)
        assert_allclose(np.dot(proj.T, proj), np.eye(dim), 1e-08, 1e-08)
        if dim == 2:
            projt = np.zeros((p, 2))
            projt[0:2, 0:2] = np.eye(2)
            assert_allclose(np.trace(np.dot(proj.T, projt)), 2, rtol=0.001, atol=0.001)