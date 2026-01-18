import re
from math import sqrt
import numpy as np
import pytest
from sklearn import metrics, neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_lof_error_n_neighbors_too_large():
    """Check that we raise a proper error message when n_neighbors == n_samples.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/17207
    """
    X = np.ones((7, 7))
    msg = 'Expected n_neighbors < n_samples_fit, but n_neighbors = 1, n_samples_fit = 1, n_samples = 1'
    with pytest.raises(ValueError, match=msg):
        lof = neighbors.LocalOutlierFactor(n_neighbors=1).fit(X[:1])
    lof = neighbors.LocalOutlierFactor(n_neighbors=2).fit(X[:2])
    assert lof.n_samples_fit_ == 2
    msg = 'Expected n_neighbors < n_samples_fit, but n_neighbors = 2, n_samples_fit = 2, n_samples = 2'
    with pytest.raises(ValueError, match=msg):
        lof.kneighbors(None, n_neighbors=2)
    distances, indices = lof.kneighbors(None, n_neighbors=1)
    assert distances.shape == (2, 1)
    assert indices.shape == (2, 1)
    msg = 'Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, n_samples_fit = 2, n_samples = 7'
    with pytest.raises(ValueError, match=msg):
        lof.kneighbors(X, n_neighbors=3)
    distances, indices = lof.kneighbors(X, n_neighbors=2)
    assert distances.shape == (7, 2)
    assert indices.shape == (7, 2)