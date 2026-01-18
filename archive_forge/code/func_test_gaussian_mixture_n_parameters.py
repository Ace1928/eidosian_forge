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
def test_gaussian_mixture_n_parameters():
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = (50, 5, 2)
    X = rng.randn(n_samples, n_features)
    n_params = {'spherical': 13, 'diag': 21, 'tied': 26, 'full': 41}
    for cv_type in COVARIANCE_TYPE:
        g = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=rng).fit(X)
        assert g._n_parameters() == n_params[cv_type]