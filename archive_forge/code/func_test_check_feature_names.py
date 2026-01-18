import numpy as np
import pytest
from sklearn.inspection._pd_utils import _check_feature_names, _get_feature_index
from sklearn.utils._testing import _convert_container
@pytest.mark.parametrize('feature_names, array_type, expected_feature_names', [(None, 'array', ['x0', 'x1', 'x2']), (None, 'dataframe', ['a', 'b', 'c']), (np.array(['a', 'b', 'c']), 'array', ['a', 'b', 'c'])])
def test_check_feature_names(feature_names, array_type, expected_feature_names):
    X = np.random.randn(10, 3)
    column_names = ['a', 'b', 'c']
    X = _convert_container(X, constructor_name=array_type, columns_name=column_names)
    feature_names_validated = _check_feature_names(X, feature_names)
    assert feature_names_validated == expected_feature_names