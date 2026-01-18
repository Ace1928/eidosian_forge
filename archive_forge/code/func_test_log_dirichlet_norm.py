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
def test_log_dirichlet_norm():
    rng = np.random.RandomState(0)
    weight_concentration = rng.rand(2)
    expected_norm = gammaln(np.sum(weight_concentration)) - np.sum(gammaln(weight_concentration))
    predected_norm = _log_dirichlet_norm(weight_concentration)
    assert_almost_equal(expected_norm, predected_norm)