import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_constant_strategy_sparse_target(csc_container):
    X = [[0]] * 5
    y = csc_container(np.array([[0, 1], [4, 0], [1, 1], [1, 4], [1, 1]]))
    n_samples = len(X)
    clf = DummyClassifier(strategy='constant', random_state=0, constant=[1, 0])
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert sp.issparse(y_pred)
    assert_array_equal(y_pred.toarray(), np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))]))