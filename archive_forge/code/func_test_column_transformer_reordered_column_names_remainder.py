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
@pytest.mark.parametrize('explicit_colname', ['first', 'second', 0, 1])
@pytest.mark.parametrize('remainder', [Trans(), 'passthrough', 'drop'])
def test_column_transformer_reordered_column_names_remainder(explicit_colname, remainder):
    """Test the interaction between remainder and column transformer"""
    pd = pytest.importorskip('pandas')
    X_fit_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_fit_df = pd.DataFrame(X_fit_array, columns=['first', 'second'])
    X_trans_array = np.array([[2, 4, 6], [0, 1, 2]]).T
    X_trans_df = pd.DataFrame(X_trans_array, columns=['second', 'first'])
    tf = ColumnTransformer([('bycol', Trans(), explicit_colname)], remainder=remainder)
    tf.fit(X_fit_df)
    X_fit_trans = tf.transform(X_fit_df)
    X_trans = tf.transform(X_trans_df)
    assert_allclose(X_trans, X_fit_trans)
    X_extended_df = X_fit_df.copy()
    X_extended_df['third'] = [3, 6, 9]
    X_trans = tf.transform(X_extended_df)
    assert_allclose(X_trans, X_fit_trans)
    if isinstance(explicit_colname, str):
        X_array = X_fit_array.copy()
        err_msg = 'Specifying the columns'
        with pytest.raises(ValueError, match=err_msg):
            tf.transform(X_array)