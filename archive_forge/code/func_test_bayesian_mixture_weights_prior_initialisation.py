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
def test_bayesian_mixture_weights_prior_initialisation():
    rng = np.random.RandomState(0)
    n_samples, n_components, n_features = (10, 5, 2)
    X = rng.rand(n_samples, n_features)
    weight_concentration_prior = rng.rand()
    bgmm = BayesianGaussianMixture(weight_concentration_prior=weight_concentration_prior, random_state=rng).fit(X)
    assert_almost_equal(weight_concentration_prior, bgmm.weight_concentration_prior_)
    bgmm = BayesianGaussianMixture(n_components=n_components, random_state=rng).fit(X)
    assert_almost_equal(1.0 / n_components, bgmm.weight_concentration_prior_)