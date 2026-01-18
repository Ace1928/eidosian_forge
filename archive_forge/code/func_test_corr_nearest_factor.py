import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
@pytest.mark.parametrize('dm', [1, 2])
def test_corr_nearest_factor(self, dm):
    objvals = [np.array([6241.8, 6241.8, 579.4, 264.6, 264.3]), np.array([2104.9, 2104.9, 710.5, 266.3, 286.1])]
    d = 100
    X = np.zeros((d, dm), dtype=np.float64)
    x = np.linspace(0, 2 * np.pi, d)
    np.random.seed(10)
    for j in range(dm):
        X[:, j] = np.sin(x * (j + 1)) + 1e-10 * np.random.randn(d)
    _project_correlation_factors(X)
    assert np.isfinite(X).all()
    X *= 0.7
    mat = np.dot(X, X.T)
    np.fill_diagonal(mat, 1.0)
    rslt = corr_nearest_factor(mat, dm, maxiter=10000)
    err_msg = 'rank=%d, niter=%d' % (dm, len(rslt.objective_values))
    assert_allclose(rslt.objective_values[:5], objvals[dm - 1], rtol=0.5, err_msg=err_msg)
    assert rslt.Converged
    mat1 = rslt.corr.to_matrix()
    assert_allclose(mat, mat1, rtol=0.25, atol=0.001, err_msg=err_msg)