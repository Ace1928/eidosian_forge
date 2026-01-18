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
@pytest.mark.parametrize('indices', [None, [1, 3]])
def test_check_method_params(indices):
    X = np.random.randn(4, 2)
    _params = {'list': [1, 2, 3, 4], 'array': np.array([1, 2, 3, 4]), 'sparse-col': sp.csc_matrix([1, 2, 3, 4]).T, 'sparse-row': sp.csc_matrix([1, 2, 3, 4]), 'scalar-int': 1, 'scalar-str': 'xxx', 'None': None}
    result = _check_method_params(X, params=_params, indices=indices)
    indices_ = indices if indices is not None else list(range(X.shape[0]))
    for key in ['sparse-row', 'scalar-int', 'scalar-str', 'None']:
        assert result[key] is _params[key]
    assert result['list'] == _safe_indexing(_params['list'], indices_)
    assert_array_equal(result['array'], _safe_indexing(_params['array'], indices_))
    assert_allclose_dense_sparse(result['sparse-col'], _safe_indexing(_params['sparse-col'], indices_))