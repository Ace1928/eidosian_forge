import numbers
import re
import warnings
from itertools import product
from operator import itemgetter
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
from pytest import importorskip
import sklearn
from sklearn._config import config_context
from sklearn._min_dependencies import dependent_packages
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError, PositiveSpectrumWarning
from sklearn.linear_model import ARDRegression
from sklearn.metrics.tests.test_score_objects import EstimatorWithFit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import _sparse_random_matrix
from sklearn.svm import SVR
from sklearn.utils import (
from sklearn.utils._mocking import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.validation import (
@pytest.mark.parametrize('value, force_all_finite', [(np.inf, False), (np.nan, 'allow-nan'), (np.nan, False)])
@pytest.mark.parametrize('retype', [np.asarray, sp.csr_matrix])
def test_check_array_force_all_finite_valid(value, force_all_finite, retype):
    X = retype(np.arange(4).reshape(2, 2).astype(float))
    X[0, 0] = value
    X_checked = check_array(X, force_all_finite=force_all_finite, accept_sparse=True)
    assert_allclose_dense_sparse(X, X_checked)