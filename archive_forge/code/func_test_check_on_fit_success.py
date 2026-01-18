import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._mocking import (
from sklearn.utils._testing import _convert_container
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('kwargs', [{}, {'check_X': _success}, {'check_y': _success}, {'check_X': _success, 'check_y': _success}])
def test_check_on_fit_success(iris, kwargs):
    X, y = iris
    CheckingClassifier(**kwargs).fit(X, y)