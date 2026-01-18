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
def test_gaussian_mixture_precisions_init_diag():
    """Check that we properly initialize `precision_cholesky_` when we manually
    provide the precision matrix.

    In this regard, we check the consistency between estimating the precision
    matrix and providing the same precision matrix as initialization. It should
    lead to the same results with the same number of iterations.

    If the initialization is wrong then the number of iterations will increase.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/16944
    """
    n_samples = 300
    rng = np.random.RandomState(0)
    shifted_gaussian = rng.randn(n_samples, 2) + np.array([20, 20])
    C = np.array([[0.0, -0.7], [3.5, 0.7]])
    stretched_gaussian = np.dot(rng.randn(n_samples, 2), C)
    X = np.vstack([shifted_gaussian, stretched_gaussian])
    n_components, covariance_type, reg_covar, random_state = (2, 'diag', 1e-06, 0)
    resp = np.zeros((X.shape[0], n_components))
    label = KMeans(n_clusters=n_components, n_init=1, random_state=random_state).fit(X).labels_
    resp[np.arange(X.shape[0]), label] = 1
    _, _, covariance = _estimate_gaussian_parameters(X, resp, reg_covar=reg_covar, covariance_type=covariance_type)
    precisions_init = 1 / covariance
    gm_with_init = GaussianMixture(n_components=n_components, covariance_type=covariance_type, reg_covar=reg_covar, precisions_init=precisions_init, random_state=random_state).fit(X)
    gm_without_init = GaussianMixture(n_components=n_components, covariance_type=covariance_type, reg_covar=reg_covar, random_state=random_state).fit(X)
    assert gm_without_init.n_iter_ == gm_with_init.n_iter_
    assert_allclose(gm_with_init.precisions_cholesky_, gm_without_init.precisions_cholesky_)