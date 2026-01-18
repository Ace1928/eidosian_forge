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
def test_gaussian_mixture_attributes():
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)
    n_components, tol, n_init, max_iter, reg_covar = (2, 0.0001, 3, 30, 0.1)
    covariance_type, init_params = ('full', 'random')
    gmm = GaussianMixture(n_components=n_components, tol=tol, n_init=n_init, max_iter=max_iter, reg_covar=reg_covar, covariance_type=covariance_type, init_params=init_params).fit(X)
    assert gmm.n_components == n_components
    assert gmm.covariance_type == covariance_type
    assert gmm.tol == tol
    assert gmm.reg_covar == reg_covar
    assert gmm.max_iter == max_iter
    assert gmm.n_init == n_init
    assert gmm.init_params == init_params