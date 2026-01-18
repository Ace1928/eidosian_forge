import numpy as np
from statsmodels.multivariate.factor import Factor
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
import warnings
def test_2factor():
    """
    # R code:
    r = 0.4
    p = 6
    ii = seq(0, p-1)
    ii = outer(ii, ii, "-")
    ii = abs(ii)
    cm = r^ii
    factanal(covmat=cm, factors=2)
    """
    r = 0.4
    p = 6
    ii = np.arange(p)
    cm = r ** np.abs(np.subtract.outer(ii, ii))
    fa = Factor(corr=cm, n_factor=2, nobs=100, method='ml')
    rslt = fa.fit()
    for j in (0, 1):
        if rslt.loadings[0, j] < 0:
            rslt.loadings[:, j] *= -1
    uniq = np.r_[0.782, 0.367, 0.696, 0.696, 0.367, 0.782]
    assert_allclose(uniq, rslt.uniqueness, rtol=0.001, atol=0.001)
    loads = [np.r_[0.323, 0.586, 0.519, 0.519, 0.586, 0.323], np.r_[0.337, 0.538, 0.187, -0.187, -0.538, -0.337]]
    for k in (0, 1):
        if np.dot(loads[k], rslt.loadings[:, k]) < 0:
            loads[k] *= -1
        assert_allclose(loads[k], rslt.loadings[:, k], rtol=0.001, atol=0.001)
    assert_equal(rslt.df, 4)
    e = np.asarray([0.11056836, 0.05191071, 0.09836349, 0.09836349, 0.05191071, 0.11056836])
    assert_allclose(rslt.uniq_stderr, e, atol=0.0001)
    e = np.asarray([[0.08842151, 0.08842151], [0.06058582, 0.06058582], [0.08339874, 0.08339874], [0.08339874, 0.08339874], [0.06058582, 0.06058582], [0.08842151, 0.08842151]])
    assert_allclose(rslt.load_stderr, e, atol=0.0001)