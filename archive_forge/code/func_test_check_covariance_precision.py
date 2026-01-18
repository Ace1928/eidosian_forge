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
@ignore_warnings(category=ConvergenceWarning)
def test_check_covariance_precision():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components, n_features = (2 * rand_data.n_components, 2)
    bgmm = BayesianGaussianMixture(n_components=n_components, max_iter=100, random_state=rng, tol=0.001, reg_covar=0)
    for covar_type in COVARIANCE_TYPE:
        bgmm.covariance_type = covar_type
        bgmm.fit(rand_data.X[covar_type])
        if covar_type == 'full':
            for covar, precision in zip(bgmm.covariances_, bgmm.precisions_):
                assert_almost_equal(np.dot(covar, precision), np.eye(n_features))
        elif covar_type == 'tied':
            assert_almost_equal(np.dot(bgmm.covariances_, bgmm.precisions_), np.eye(n_features))
        elif covar_type == 'diag':
            assert_almost_equal(bgmm.covariances_ * bgmm.precisions_, np.ones((n_components, n_features)))
        else:
            assert_almost_equal(bgmm.covariances_ * bgmm.precisions_, np.ones(n_components))