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
def test_column_transformer_callable_specifier_dataframe():
    pd = pytest.importorskip('pandas')
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_res_first = np.array([[0, 1, 2]]).T
    X_df = pd.DataFrame(X_array, columns=['first', 'second'])

    def func(X):
        assert_array_equal(X.columns, X_df.columns)
        assert_array_equal(X.values, X_df.values)
        return ['first']
    ct = ColumnTransformer([('trans', Trans(), func)], remainder='drop')
    assert_array_equal(ct.fit_transform(X_df), X_res_first)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_first)
    assert callable(ct.transformers[0][2])
    assert ct.transformers_[0][2] == ['first']