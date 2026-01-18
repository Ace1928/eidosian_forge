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
def test_bayesian_mixture_check_is_fitted():
    rng = np.random.RandomState(0)
    n_samples, n_features = (10, 2)
    bgmm = BayesianGaussianMixture(random_state=rng)
    X = rng.rand(n_samples, n_features)
    msg = 'This BayesianGaussianMixture instance is not fitted yet.'
    with pytest.raises(ValueError, match=msg):
        bgmm.score(X)