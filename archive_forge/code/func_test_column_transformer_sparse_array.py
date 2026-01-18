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
def test_column_transformer_sparse_array(csr_container):
    X_sparse = csr_container(sparse.eye(3, 2))
    X_res_first = X_sparse[:, [0]]
    X_res_both = X_sparse
    for col in [(0,), [0], slice(0, 1)]:
        for remainder, res in [('drop', X_res_first), ('passthrough', X_res_both)]:
            ct = ColumnTransformer([('trans', Trans(), col)], remainder=remainder, sparse_threshold=0.8)
            assert sparse.issparse(ct.fit_transform(X_sparse))
            assert_allclose_dense_sparse(ct.fit_transform(X_sparse), res)
            assert_allclose_dense_sparse(ct.fit(X_sparse).transform(X_sparse), res)
    for col in [[0, 1], slice(0, 2)]:
        ct = ColumnTransformer([('trans', Trans(), col)], sparse_threshold=0.8)
        assert sparse.issparse(ct.fit_transform(X_sparse))
        assert_allclose_dense_sparse(ct.fit_transform(X_sparse), X_res_both)
        assert_allclose_dense_sparse(ct.fit(X_sparse).transform(X_sparse), X_res_both)