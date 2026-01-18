import copy
import itertools
import re
import sys
import warnings
from io import StringIO
from unittest.mock import Mock
import numpy as np
import pytest
from scipy import linalg, stats
import sklearn
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_spd_matrix
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_gaussian_mixture_estimate_log_prob_resp():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=5)
    n_samples = rand_data.n_samples
    n_features = rand_data.n_features
    n_components = rand_data.n_components
    X = rng.rand(n_samples, n_features)
    for covar_type in COVARIANCE_TYPE:
        weights = rand_data.weights
        means = rand_data.means
        precisions = rand_data.precisions[covar_type]
        g = GaussianMixture(n_components=n_components, random_state=rng, weights_init=weights, means_init=means, precisions_init=precisions, covariance_type=covar_type)
        g.fit(X)
        resp = g.predict_proba(X)
        assert_array_almost_equal(resp.sum(axis=1), np.ones(n_samples))
        assert_array_equal(g.weights_init, weights)
        assert_array_equal(g.means_init, means)
        assert_array_equal(g.precisions_init, precisions)