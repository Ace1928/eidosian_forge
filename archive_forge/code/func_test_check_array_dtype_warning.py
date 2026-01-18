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
def test_check_array_dtype_warning():
    X_int_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    X_float32 = np.asarray(X_int_list, dtype=np.float32)
    X_int64 = np.asarray(X_int_list, dtype=np.int64)
    X_csr_float32 = sp.csr_matrix(X_float32)
    X_csc_float32 = sp.csc_matrix(X_float32)
    X_csc_int32 = sp.csc_matrix(X_int64, dtype=np.int32)
    integer_data = [X_int64, X_csc_int32]
    float32_data = [X_float32, X_csr_float32, X_csc_float32]
    for X in integer_data:
        X_checked = assert_no_warnings(check_array, X, dtype=np.float64, accept_sparse=True)
        assert X_checked.dtype == np.float64
    for X in float32_data:
        X_checked = assert_no_warnings(check_array, X, dtype=[np.float64, np.float32], accept_sparse=True)
        assert X_checked.dtype == np.float32
        assert X_checked is X
        X_checked = assert_no_warnings(check_array, X, dtype=[np.float64, np.float32], accept_sparse=['csr', 'dok'], copy=True)
        assert X_checked.dtype == np.float32
        assert X_checked is not X
    X_checked = assert_no_warnings(check_array, X_csc_float32, dtype=[np.float64, np.float32], accept_sparse=['csr', 'dok'], copy=False)
    assert X_checked.dtype == np.float32
    assert X_checked is not X_csc_float32
    assert X_checked.format == 'csr'