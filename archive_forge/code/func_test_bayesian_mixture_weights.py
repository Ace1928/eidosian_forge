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
def test_bayesian_mixture_weights():
    rng = np.random.RandomState(0)
    n_samples, n_features = (10, 2)
    X = rng.rand(n_samples, n_features)
    bgmm = BayesianGaussianMixture(weight_concentration_prior_type='dirichlet_distribution', n_components=3, random_state=rng).fit(X)
    expected_weights = bgmm.weight_concentration_ / np.sum(bgmm.weight_concentration_)
    assert_almost_equal(expected_weights, bgmm.weights_)
    assert_almost_equal(np.sum(bgmm.weights_), 1.0)
    dpgmm = BayesianGaussianMixture(weight_concentration_prior_type='dirichlet_process', n_components=3, random_state=rng).fit(X)
    weight_dirichlet_sum = dpgmm.weight_concentration_[0] + dpgmm.weight_concentration_[1]
    tmp = dpgmm.weight_concentration_[1] / weight_dirichlet_sum
    expected_weights = dpgmm.weight_concentration_[0] / weight_dirichlet_sum * np.hstack((1, np.cumprod(tmp[:-1])))
    expected_weights /= np.sum(expected_weights)
    assert_almost_equal(expected_weights, dpgmm.weights_)
    assert_almost_equal(np.sum(dpgmm.weights_), 1.0)