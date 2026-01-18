import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
def test_fdr_twostage():
    pvals = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.324, 0.4262, 0.5719, 0.6528, 0.759, 1.0]
    n = len(pvals)
    k = 0
    res0 = multipletests(pvals, alpha=0.05, method='fdr_bh')
    res1 = fdrcorrection_twostage(pvals, alpha=0.05, method='bh', maxiter=k, iter=None)
    res2 = multipletests(pvals, alpha=0.05, method='fdr_tsbh', maxiter=k)
    assert_allclose(res1[1], res0[1])
    assert_allclose(res2[1], res1[1])
    k = 1
    res0 = multipletests(pvals, alpha=0.05, method='fdr_bh')
    res1 = fdrcorrection_twostage(pvals, alpha=0.05, method='bh', maxiter=k, iter=None)
    res2 = multipletests(pvals, alpha=0.05, method='fdr_tsbh', maxiter=k)
    res3 = multipletests(pvals, alpha=0.05, method='fdr_tsbh')
    assert_allclose(res1[1], res0[1] * (1 - res0[0].sum() / n))
    assert_allclose(res2[1], res1[1])
    assert_allclose(res3[1], res1[1])
    fact = 1 + 0.05
    k = 0
    res0 = multipletests(pvals, alpha=0.05, method='fdr_bh')
    res1 = fdrcorrection_twostage(pvals, alpha=0.05, method='bky', maxiter=k, iter=None)
    res2 = multipletests(pvals, alpha=0.05, method='fdr_tsbky', maxiter=k)
    assert_allclose(res1[1], np.clip(res0[1] * fact, 0, 1))
    assert_allclose(res2[1], res1[1])
    k = 1
    res0 = multipletests(pvals, alpha=0.05, method='fdr_bh')
    res1 = fdrcorrection_twostage(pvals, alpha=0.05, method='bky', maxiter=k, iter=None)
    res2 = multipletests(pvals, alpha=0.05, method='fdr_tsbky', maxiter=k)
    res3 = multipletests(pvals, alpha=0.05, method='fdr_tsbky')
    assert_allclose(res1[1], res0[1] * (1 - res0[0].sum() / n) * fact)
    assert_allclose(res2[1], res1[1])
    assert_allclose(res3[1], res1[1])