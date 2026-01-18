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
def test_make_column_selector_pickle():
    pd = pytest.importorskip('pandas')
    X_df = pd.DataFrame({'col_int': np.array([0, 1, 2], dtype=int), 'col_float': np.array([0.0, 1.0, 2.0], dtype=float), 'col_str': ['one', 'two', 'three']}, columns=['col_int', 'col_float', 'col_str'])
    selector = make_column_selector(dtype_include=[object])
    selector_picked = pickle.loads(pickle.dumps(selector))
    assert_array_equal(selector(X_df), selector_picked(X_df))