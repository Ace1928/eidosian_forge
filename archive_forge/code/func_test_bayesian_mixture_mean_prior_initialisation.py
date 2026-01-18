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
def test_bayesian_mixture_mean_prior_initialisation():
    rng = np.random.RandomState(0)
    n_samples, n_components, n_features = (10, 3, 2)
    X = rng.rand(n_samples, n_features)
    mean_precision_prior = rng.rand()
    bgmm = BayesianGaussianMixture(mean_precision_prior=mean_precision_prior, random_state=rng).fit(X)
    assert_almost_equal(mean_precision_prior, bgmm.mean_precision_prior_)
    bgmm = BayesianGaussianMixture(random_state=rng).fit(X)
    assert_almost_equal(1.0, bgmm.mean_precision_prior_)
    mean_prior = rng.rand(n_features)
    bgmm = BayesianGaussianMixture(n_components=n_components, mean_prior=mean_prior, random_state=rng).fit(X)
    assert_almost_equal(mean_prior, bgmm.mean_prior_)
    bgmm = BayesianGaussianMixture(n_components=n_components, random_state=rng).fit(X)
    assert_almost_equal(X.mean(axis=0), bgmm.mean_prior_)