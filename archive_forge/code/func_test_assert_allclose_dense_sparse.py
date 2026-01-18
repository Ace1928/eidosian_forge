import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
@pytest.mark.parametrize('csr_container', CSC_CONTAINERS)
def test_assert_allclose_dense_sparse(csr_container):
    x = np.arange(9).reshape(3, 3)
    msg = 'Not equal to tolerance '
    y = csr_container(x)
    for X in [x, y]:
        with pytest.raises(AssertionError, match=msg):
            assert_allclose_dense_sparse(X, X * 2)
        assert_allclose_dense_sparse(X, X)
    with pytest.raises(ValueError, match='Can only compare two sparse'):
        assert_allclose_dense_sparse(x, y)
    A = sparse.diags(np.ones(5), offsets=0).tocsr()
    B = csr_container(np.ones((1, 5)))
    with pytest.raises(AssertionError, match='Arrays are not equal'):
        assert_allclose_dense_sparse(B, A)