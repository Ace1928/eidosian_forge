import numpy as np
import pytest
from sklearn.inspection._pd_utils import _check_feature_names, _get_feature_index
from sklearn.utils._testing import _convert_container
@pytest.mark.parametrize('fx, feature_names, err_msg', [('a', None, "Cannot plot partial dependence for feature 'a'"), ('d', ['a', 'b', 'c'], "Feature 'd' not in feature_names")])
def test_get_feature_names_error(fx, feature_names, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        _get_feature_index(fx, feature_names)