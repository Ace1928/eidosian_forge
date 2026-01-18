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
def test_check_array_accept_sparse_type_exception():
    X = [[1, 2], [3, 4]]
    X_csr = sp.csr_matrix(X)
    invalid_type = SVR()
    msg = "Sparse data was passed, but dense data is required. Use '.toarray\\(\\)' to convert to a dense numpy array."
    with pytest.raises(TypeError, match=msg):
        check_array(X_csr, accept_sparse=False)
    msg = "Parameter 'accept_sparse' should be a string, boolean or list of strings. You provided 'accept_sparse=.*'."
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=invalid_type)
    msg = "When providing 'accept_sparse' as a tuple or list, it must contain at least one string value."
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=[])
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=())
    with pytest.raises(TypeError, match='SVR'):
        check_array(X_csr, accept_sparse=[invalid_type])