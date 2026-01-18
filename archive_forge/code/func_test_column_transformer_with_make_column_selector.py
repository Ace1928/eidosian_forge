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
def test_column_transformer_with_make_column_selector():
    pd = pytest.importorskip('pandas')
    X_df = pd.DataFrame({'col_int': np.array([0, 1, 2], dtype=int), 'col_float': np.array([0.0, 1.0, 2.0], dtype=float), 'col_cat': ['one', 'two', 'one'], 'col_str': ['low', 'middle', 'high']}, columns=['col_int', 'col_float', 'col_cat', 'col_str'])
    X_df['col_str'] = X_df['col_str'].astype('category')
    cat_selector = make_column_selector(dtype_include=['category', object])
    num_selector = make_column_selector(dtype_include=np.number)
    ohe = OneHotEncoder()
    scaler = StandardScaler()
    ct_selector = make_column_transformer((ohe, cat_selector), (scaler, num_selector))
    ct_direct = make_column_transformer((ohe, ['col_cat', 'col_str']), (scaler, ['col_float', 'col_int']))
    X_selector = ct_selector.fit_transform(X_df)
    X_direct = ct_direct.fit_transform(X_df)
    assert_allclose(X_selector, X_direct)