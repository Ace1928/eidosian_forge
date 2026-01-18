import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
def test_fdr_bky():
    pvals = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.324, 0.4262, 0.5719, 0.6528, 0.759, 1.0]
    with pytest.warns(FutureWarning, match='iter keyword'):
        res_tst = fdrcorrection_twostage(pvals, alpha=0.05, iter=False)
    assert_almost_equal([0.047619, 0.0649], res_tst[-1][:2], 3)
    assert_equal(8, res_tst[0].sum())
    res2 = np.array([0.0012, 0.0023, 0.0073, 0.0274, 0.0464, 0.0492, 0.0492, 0.0497, 0.0589, 0.3742, 0.4475, 0.5505, 0.58, 0.6262, 0.77])
    assert_allclose(res_tst[1], res2, atol=6e-05)
    pvals = np.array([0.2, 0.8, 0.3, 0.5, 1])
    res1 = fdrcorrection_twostage(pvals, alpha=0.05, method='bky')
    res2 = multipletests(pvals, alpha=0.05, method='fdr_tsbky')
    assert_equal(res1[0], res2[0])
    assert_allclose(res1[1], res2[1], atol=6e-05)
    res_pv = np.array([0.7875, 1.0, 0.7875, 0.875, 1.0])
    assert_allclose(res1[1], res_pv, atol=6e-05)