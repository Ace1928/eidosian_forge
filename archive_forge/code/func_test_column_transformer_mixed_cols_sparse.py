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
def test_column_transformer_mixed_cols_sparse():
    df = np.array([['a', 1, True], ['b', 2, False]], dtype='O')
    ct = make_column_transformer((OneHotEncoder(), [0]), ('passthrough', [1, 2]), sparse_threshold=1.0)
    X_trans = ct.fit_transform(df)
    assert X_trans.getformat() == 'csr'
    assert_array_equal(X_trans.toarray(), np.array([[1, 0, 1, 1], [0, 1, 2, 0]]))
    ct = make_column_transformer((OneHotEncoder(), [0]), ('passthrough', [0]), sparse_threshold=1.0)
    with pytest.raises(ValueError, match='For a sparse output, all columns should'):
        ct.fit_transform(df)