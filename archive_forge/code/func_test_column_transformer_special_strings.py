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
def test_column_transformer_special_strings():
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    ct = ColumnTransformer([('trans1', Trans(), [0]), ('trans2', 'drop', [1])])
    exp = np.array([[0.0], [1.0], [2.0]])
    assert_array_equal(ct.fit_transform(X_array), exp)
    assert_array_equal(ct.fit(X_array).transform(X_array), exp)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] != 'remainder'
    ct = ColumnTransformer([('trans1', 'drop', [0]), ('trans2', 'drop', [1])])
    assert_array_equal(ct.fit(X_array).transform(X_array).shape, (3, 0))
    assert_array_equal(ct.fit_transform(X_array).shape, (3, 0))
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] != 'remainder'
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    ct = ColumnTransformer([('trans1', Trans(), [0]), ('trans2', 'passthrough', [1])])
    exp = X_array
    assert_array_equal(ct.fit_transform(X_array), exp)
    assert_array_equal(ct.fit(X_array).transform(X_array), exp)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] != 'remainder'