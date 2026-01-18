import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._mocking import (
from sklearn.utils._testing import _convert_container
from sklearn.utils.fixes import CSR_CONTAINERS
def test_checking_classifier_fit_params(iris):
    X, y = iris
    clf = CheckingClassifier(expected_sample_weight=True)
    sample_weight = np.ones(len(X) // 2)
    msg = f'sample_weight.shape == ({len(X) // 2},), expected ({len(X)},)!'
    with pytest.raises(ValueError) as exc:
        clf.fit(X, y, sample_weight=sample_weight)
    assert exc.value.args[0] == msg