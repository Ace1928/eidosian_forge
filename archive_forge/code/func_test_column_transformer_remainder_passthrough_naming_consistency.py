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
@pytest.mark.parametrize('transform_output', ['default', 'pandas'])
def test_column_transformer_remainder_passthrough_naming_consistency(transform_output):
    """Check that when `remainder="passthrough"`, inconsistent naming is handled
    correctly by the underlying `FunctionTransformer`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28232
    """
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame(np.random.randn(10, 4))
    preprocessor = ColumnTransformer(transformers=[('scaler', StandardScaler(), [0, 1])], remainder='passthrough').set_output(transform=transform_output)
    X_trans = preprocessor.fit_transform(X)
    assert X_trans.shape == X.shape
    expected_column_names = ['scaler__x0', 'scaler__x1', 'remainder__x2', 'remainder__x3']
    if hasattr(X_trans, 'columns'):
        assert X_trans.columns.tolist() == expected_column_names
    assert preprocessor.get_feature_names_out().tolist() == expected_column_names