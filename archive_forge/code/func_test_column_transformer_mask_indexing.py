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
@pytest.mark.parametrize('array_type', [np.asarray, *CSR_CONTAINERS])
def test_column_transformer_mask_indexing(array_type):
    X = np.transpose([[1, 2, 3], [4, 5, 6], [5, 6, 7], [8, 9, 10]])
    X = array_type(X)
    column_transformer = ColumnTransformer([('identity', FunctionTransformer(), [False, True, False, True])])
    X_trans = column_transformer.fit_transform(X)
    assert X_trans.shape == (3, 2)