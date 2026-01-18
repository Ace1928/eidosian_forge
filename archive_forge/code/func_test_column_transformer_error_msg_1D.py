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
def test_column_transformer_error_msg_1D():
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    col_trans = ColumnTransformer([('trans', StandardScaler(), 0)])
    msg = '1D data passed to a transformer'
    with pytest.raises(ValueError, match=msg):
        col_trans.fit(X_array)
    with pytest.raises(ValueError, match=msg):
        col_trans.fit_transform(X_array)
    col_trans = ColumnTransformer([('trans', TransRaise(), 0)])
    for func in [col_trans.fit, col_trans.fit_transform]:
        with pytest.raises(ValueError, match='specific message'):
            func(X_array)