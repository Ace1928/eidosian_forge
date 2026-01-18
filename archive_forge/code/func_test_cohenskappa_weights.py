import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from statsmodels.stats.inter_rater import (fleiss_kappa, cohens_kappa,
from statsmodels.tools.testing import Holder
def test_cohenskappa_weights():
    np.random.seed(9743678)
    table = np.random.randint(0, 10, size=(5, 5)) + 5 * np.eye(5)
    mat = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
    table_agg = np.dot(np.dot(mat, table), mat.T)
    res1 = cohens_kappa(table, weights=np.arange(5) > 2, wt='linear')
    res2 = cohens_kappa(table_agg, weights=np.arange(2), wt='linear')
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)
    res1 = cohens_kappa(table, weights=2 * np.arange(5), wt='linear')
    res2 = cohens_kappa(table, weights=2 * np.arange(5), wt='toeplitz')
    res3 = cohens_kappa(table, weights=res1.weights[0], wt='toeplitz')
    res4 = cohens_kappa(table, weights=res1.weights)
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)
    assert_almost_equal(res1.kappa, res3.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res3.var_kappa, decimal=14)
    assert_almost_equal(res1.kappa, res4.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res4.var_kappa, decimal=14)
    res1 = cohens_kappa(table, weights=5 * np.arange(5) ** 2, wt='toeplitz')
    res2 = cohens_kappa(table, weights=5 * np.arange(5), wt='quadratic')
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)