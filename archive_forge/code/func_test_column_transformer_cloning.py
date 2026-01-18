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
def test_column_transformer_cloning():
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    ct = ColumnTransformer([('trans', StandardScaler(), [0])])
    ct.fit(X_array)
    assert not hasattr(ct.transformers[0][1], 'mean_')
    assert hasattr(ct.transformers_[0][1], 'mean_')
    ct = ColumnTransformer([('trans', StandardScaler(), [0])])
    ct.fit_transform(X_array)
    assert not hasattr(ct.transformers[0][1], 'mean_')
    assert hasattr(ct.transformers_[0][1], 'mean_')