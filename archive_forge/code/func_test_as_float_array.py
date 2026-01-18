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
def test_as_float_array():
    X = np.ones((3, 10), dtype=np.int32)
    X = X + np.arange(10, dtype=np.int32)
    X2 = as_float_array(X, copy=False)
    assert X2.dtype == np.float32
    X = X.astype(np.int64)
    X2 = as_float_array(X, copy=True)
    assert as_float_array(X, copy=False) is not X
    assert X2.dtype == np.float64
    tested_dtypes = [bool, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32]
    for dtype in tested_dtypes:
        X = X.astype(dtype)
        X2 = as_float_array(X)
        assert X2.dtype == np.float32
    X = X.astype(object)
    X2 = as_float_array(X, copy=True)
    assert X2.dtype == np.float64
    X = np.ones((3, 2), dtype=np.float32)
    assert as_float_array(X, copy=False) is X
    X = np.asfortranarray(X)
    assert np.isfortran(as_float_array(X, copy=True))
    matrices = [sp.csc_matrix(np.arange(5)).toarray(), _sparse_random_matrix(10, 10, density=0.1).toarray()]
    for M in matrices:
        N = as_float_array(M, copy=True)
        N[0, 0] = np.nan
        assert not np.isnan(M).any()