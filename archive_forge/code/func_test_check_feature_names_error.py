import numpy as np
import pytest
from sklearn.inspection._pd_utils import _check_feature_names, _get_feature_index
from sklearn.utils._testing import _convert_container
def test_check_feature_names_error():
    X = np.random.randn(10, 3)
    feature_names = ['a', 'b', 'c', 'a']
    msg = 'feature_names should not contain duplicates.'
    with pytest.raises(ValueError, match=msg):
        _check_feature_names(X, feature_names)