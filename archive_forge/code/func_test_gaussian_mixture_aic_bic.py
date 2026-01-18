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
def test_gaussian_mixture_aic_bic():
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = (50, 3, 2)
    X = rng.randn(n_samples, n_features)
    sgh = 0.5 * (fast_logdet(np.cov(X.T, bias=1)) + n_features * (1 + np.log(2 * np.pi)))
    for cv_type in COVARIANCE_TYPE:
        g = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=rng, max_iter=200)
        g.fit(X)
        aic = 2 * n_samples * sgh + 2 * g._n_parameters()
        bic = 2 * n_samples * sgh + np.log(n_samples) * g._n_parameters()
        bound = n_features / np.sqrt(n_samples)
        assert (g.aic(X) - aic) / n_samples < bound
        assert (g.bic(X) - bic) / n_samples < bound