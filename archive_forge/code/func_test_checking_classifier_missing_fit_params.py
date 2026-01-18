import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._mocking import (
from sklearn.utils._testing import _convert_container
from sklearn.utils.fixes import CSR_CONTAINERS
def test_checking_classifier_missing_fit_params(iris):
    X, y = iris
    clf = CheckingClassifier(expected_sample_weight=True)
    err_msg = 'Expected sample_weight to be passed'
    with pytest.raises(AssertionError, match=err_msg):
        clf.fit(X, y)