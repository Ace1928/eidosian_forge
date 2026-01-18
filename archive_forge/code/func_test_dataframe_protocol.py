import pickle
import re
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
import sklearn
from sklearn import config_context, datasets
from sklearn.base import (
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._set_output import _get_output_config
from sklearn.utils._testing import (
@pytest.mark.parametrize('constructor_name, minversion', [('dataframe', '1.5.0'), ('pyarrow', '12.0.0'), ('polars', '0.19.12')])
def test_dataframe_protocol(constructor_name, minversion):
    """Uses the dataframe exchange protocol to get feature names."""
    data = [[1, 4, 2], [3, 3, 6]]
    columns = ['col_0', 'col_1', 'col_2']
    df = _convert_container(data, constructor_name, columns_name=columns, minversion=minversion)

    class NoOpTransformer(TransformerMixin, BaseEstimator):

        def fit(self, X, y=None):
            self._validate_data(X)
            return self

        def transform(self, X):
            return self._validate_data(X, reset=False)
    no_op = NoOpTransformer()
    no_op.fit(df)
    assert_array_equal(no_op.feature_names_in_, columns)
    X_out = no_op.transform(df)
    if constructor_name != 'pyarrow':
        assert_allclose(df, X_out)
    bad_names = ['a', 'b', 'c']
    df_bad = _convert_container(data, constructor_name, columns_name=bad_names)
    with pytest.raises(ValueError, match='The feature names should match'):
        no_op.transform(df_bad)