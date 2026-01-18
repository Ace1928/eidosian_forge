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
def test_column_transformer_no_remaining_remainder_transformer():
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T
    ct = ColumnTransformer([('trans1', Trans(), [0, 1, 2])], remainder=DoubleTrans())
    assert_array_equal(ct.fit_transform(X_array), X_array)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_array)
    assert len(ct.transformers_) == 1
    assert ct.transformers_[-1][0] != 'remainder'