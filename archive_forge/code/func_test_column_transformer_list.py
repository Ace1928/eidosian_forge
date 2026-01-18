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
def test_column_transformer_list():
    X_list = [[1, float('nan'), 'a'], [0, 0, 'b']]
    expected_result = np.array([[1, float('nan'), 1, 0], [-1, 0, 0, 1]])
    ct = ColumnTransformer([('numerical', StandardScaler(), [0, 1]), ('categorical', OneHotEncoder(), [2])])
    assert_array_equal(ct.fit_transform(X_list), expected_result)
    assert_array_equal(ct.fit(X_list).transform(X_list), expected_result)