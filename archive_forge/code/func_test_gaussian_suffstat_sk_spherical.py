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
def test_gaussian_suffstat_sk_spherical():
    rng = np.random.RandomState(0)
    n_samples, n_features = (500, 2)
    X = rng.rand(n_samples, n_features)
    X = X - X.mean()
    resp = np.ones((n_samples, 1))
    nk = np.array([n_samples])
    xk = X.mean()
    covars_pred_spherical = _estimate_gaussian_covariances_spherical(resp, X, nk, xk, 0)
    covars_pred_spherical2 = np.dot(X.flatten().T, X.flatten()) / (n_features * n_samples)
    assert_almost_equal(covars_pred_spherical, covars_pred_spherical2)
    precs_chol_pred = _compute_precision_cholesky(covars_pred_spherical, 'spherical')
    assert_almost_equal(covars_pred_spherical, 1.0 / precs_chol_pred ** 2)