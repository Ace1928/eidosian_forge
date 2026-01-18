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
def test_column_transformer_remainder():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_res_first = np.array([0, 1, 2]).reshape(-1, 1)
    X_res_second = np.array([2, 4, 6]).reshape(-1, 1)
    X_res_both = X_array
    ct = ColumnTransformer([('trans1', Trans(), [0])])
    assert_array_equal(ct.fit_transform(X_array), X_res_first)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_first)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == 'remainder'
    assert ct.transformers_[-1][1] == 'drop'
    assert_array_equal(ct.transformers_[-1][2], [1])
    ct = ColumnTransformer([('trans', Trans(), [0])], remainder='passthrough')
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == 'remainder'
    assert isinstance(ct.transformers_[-1][1], FunctionTransformer)
    assert_array_equal(ct.transformers_[-1][2], [1])
    ct = ColumnTransformer([('trans1', Trans(), [1])], remainder='passthrough')
    assert_array_equal(ct.fit_transform(X_array), X_res_both[:, ::-1])
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both[:, ::-1])
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == 'remainder'
    assert isinstance(ct.transformers_[-1][1], FunctionTransformer)
    assert_array_equal(ct.transformers_[-1][2], [0])
    ct = ColumnTransformer([('trans1', 'drop', [0])], remainder='passthrough')
    assert_array_equal(ct.fit_transform(X_array), X_res_second)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_second)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == 'remainder'
    assert isinstance(ct.transformers_[-1][1], FunctionTransformer)
    assert_array_equal(ct.transformers_[-1][2], [1])
    ct = make_column_transformer((Trans(), [0]))
    assert ct.remainder == 'drop'