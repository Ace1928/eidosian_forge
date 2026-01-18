import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
@pytest.mark.parametrize('constructor', [np.array, list] + CSC_CONTAINERS + CSR_CONTAINERS)
def test_binarizer(constructor):
    X_ = np.array([[1, 0, 5], [2, 3, -1]])
    X = constructor(X_.copy())
    binarizer = Binarizer(threshold=2.0, copy=True)
    X_bin = toarray(binarizer.transform(X))
    assert np.sum(X_bin == 0) == 4
    assert np.sum(X_bin == 1) == 2
    X_bin = binarizer.transform(X)
    assert sparse.issparse(X) == sparse.issparse(X_bin)
    binarizer = Binarizer(copy=True).fit(X)
    X_bin = toarray(binarizer.transform(X))
    assert X_bin is not X
    assert np.sum(X_bin == 0) == 2
    assert np.sum(X_bin == 1) == 4
    binarizer = Binarizer(copy=True)
    X_bin = binarizer.transform(X)
    assert X_bin is not X
    X_bin = toarray(X_bin)
    assert np.sum(X_bin == 0) == 2
    assert np.sum(X_bin == 1) == 4
    binarizer = Binarizer(copy=False)
    X_bin = binarizer.transform(X)
    if constructor is not list:
        assert X_bin is X
    binarizer = Binarizer(copy=False)
    X_float = np.array([[1, 0, 5], [2, 3, -1]], dtype=np.float64)
    X_bin = binarizer.transform(X_float)
    if constructor is not list:
        assert X_bin is X_float
    X_bin = toarray(X_bin)
    assert np.sum(X_bin == 0) == 2
    assert np.sum(X_bin == 1) == 4
    binarizer = Binarizer(threshold=-0.5, copy=True)
    if constructor in (np.array, list):
        X = constructor(X_.copy())
        X_bin = toarray(binarizer.transform(X))
        assert np.sum(X_bin == 0) == 1
        assert np.sum(X_bin == 1) == 5
        X_bin = binarizer.transform(X)
    if constructor in CSC_CONTAINERS:
        with pytest.raises(ValueError):
            binarizer.transform(constructor(X))