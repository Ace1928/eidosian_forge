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
def test_make_column_transformer_pandas():
    pd = pytest.importorskip('pandas')
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = pd.DataFrame(X_array, columns=['first', 'second'])
    norm = Normalizer()
    ct1 = ColumnTransformer([('norm', Normalizer(), X_df.columns)])
    ct2 = make_column_transformer((norm, X_df.columns))
    assert_almost_equal(ct1.fit_transform(X_df), ct2.fit_transform(X_df))