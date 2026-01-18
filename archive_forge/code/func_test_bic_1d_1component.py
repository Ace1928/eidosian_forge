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
def test_bic_1d_1component():
    rng = np.random.RandomState(0)
    n_samples, n_dim, n_components = (100, 1, 1)
    X = rng.randn(n_samples, n_dim)
    bic_full = GaussianMixture(n_components=n_components, covariance_type='full', random_state=rng).fit(X).bic(X)
    for covariance_type in ['tied', 'diag', 'spherical']:
        bic = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=rng).fit(X).bic(X)
        assert_almost_equal(bic_full, bic)