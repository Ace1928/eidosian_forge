import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_column_transformer_sparse_stacking(csr_container):
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    col_trans = ColumnTransformer([('trans1', Trans(), [0]), ('trans2', SparseMatrixTrans(csr_container), 1)], sparse_threshold=0.8)
    col_trans.fit(X_array)
    X_trans = col_trans.transform(X_array)
    assert sparse.issparse(X_trans)
    assert X_trans.shape == (X_trans.shape[0], X_trans.shape[0] + 1)
    assert_array_equal(X_trans.toarray()[:, 1:], np.eye(X_trans.shape[0]))
    assert len(col_trans.transformers_) == 2
    assert col_trans.transformers_[-1][0] != 'remainder'
    col_trans = ColumnTransformer([('trans1', Trans(), [0]), ('trans2', SparseMatrixTrans(csr_container), 1)], sparse_threshold=0.1)
    col_trans.fit(X_array)
    X_trans = col_trans.transform(X_array)
    assert not sparse.issparse(X_trans)
    assert X_trans.shape == (X_trans.shape[0], X_trans.shape[0] + 1)
    assert_array_equal(X_trans[:, 1:], np.eye(X_trans.shape[0]))