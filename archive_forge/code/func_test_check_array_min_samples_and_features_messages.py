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
def test_check_array_min_samples_and_features_messages():
    msg = '0 feature\\(s\\) \\(shape=\\(1, 0\\)\\) while a minimum of 1 is required.'
    with pytest.raises(ValueError, match=msg):
        check_array([[]])
    msg = '0 sample\\(s\\) \\(shape=\\(0,\\)\\) while a minimum of 1 is required.'
    with pytest.raises(ValueError, match=msg):
        check_array([], ensure_2d=False)
    msg = 'Singleton array array\\(42\\) cannot be considered a valid collection.'
    with pytest.raises(TypeError, match=msg):
        check_array(42, ensure_2d=False)
    X = np.ones((1, 10))
    y = np.ones(1)
    msg = '1 sample\\(s\\) \\(shape=\\(1, 10\\)\\) while a minimum of 2 is required.'
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_samples=2)
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_samples=2, ensure_2d=False)
    X = np.ones((10, 2))
    y = np.ones(2)
    msg = '2 feature\\(s\\) \\(shape=\\(10, 2\\)\\) while a minimum of 3 is required.'
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_features=3)
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_features=3, allow_nd=True)
    X = np.empty(0).reshape(10, 0)
    y = np.ones(10)
    msg = '0 feature\\(s\\) \\(shape=\\(10, 0\\)\\) while a minimum of 1 is required.'
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y)
    X = np.ones((10, 0, 28, 28))
    y = np.ones(10)
    X_checked, y_checked = check_X_y(X, y, allow_nd=True)
    assert_array_equal(X, X_checked)
    assert_array_equal(y, y_checked)