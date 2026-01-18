import copy
import numpy as np
import pytest
from scipy.special import gammaln
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture._bayesian_mixture import _log_dirichlet_norm, _log_wishart_norm
from sklearn.mixture.tests.test_gaussian_mixture import RandomData
from sklearn.utils._testing import (
@pytest.mark.filterwarnings('ignore:.*did not converge.*')
@pytest.mark.parametrize('seed, max_iter, tol', [(0, 2, 1e-07), (1, 2, 0.1), (3, 300, 1e-07), (4, 300, 0.1)])
def test_bayesian_mixture_fit_predict(seed, max_iter, tol):
    rng = np.random.RandomState(seed)
    rand_data = RandomData(rng, n_samples=50, scale=7)
    n_components = 2 * rand_data.n_components
    for covar_type in COVARIANCE_TYPE:
        bgmm1 = BayesianGaussianMixture(n_components=n_components, max_iter=max_iter, random_state=rng, tol=tol, reg_covar=0)
        bgmm1.covariance_type = covar_type
        bgmm2 = copy.deepcopy(bgmm1)
        X = rand_data.X[covar_type]
        Y_pred1 = bgmm1.fit(X).predict(X)
        Y_pred2 = bgmm2.fit_predict(X)
        assert_array_equal(Y_pred1, Y_pred2)