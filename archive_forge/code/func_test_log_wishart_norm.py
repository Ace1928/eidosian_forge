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
def test_log_wishart_norm():
    rng = np.random.RandomState(0)
    n_components, n_features = (5, 2)
    degrees_of_freedom = np.abs(rng.rand(n_components)) + 1.0
    log_det_precisions_chol = n_features * np.log(range(2, 2 + n_components))
    expected_norm = np.empty(5)
    for k, (degrees_of_freedom_k, log_det_k) in enumerate(zip(degrees_of_freedom, log_det_precisions_chol)):
        expected_norm[k] = -(degrees_of_freedom_k * (log_det_k + 0.5 * n_features * np.log(2.0)) + np.sum(gammaln(0.5 * (degrees_of_freedom_k - np.arange(0, n_features)[:, np.newaxis])), 0)).item()
    predected_norm = _log_wishart_norm(degrees_of_freedom, log_det_precisions_chol, n_features)
    assert_almost_equal(expected_norm, predected_norm)