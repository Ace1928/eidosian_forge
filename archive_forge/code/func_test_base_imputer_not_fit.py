import numpy as np
import pytest
from sklearn.impute._base import _BaseImputer
from sklearn.impute._iterative import _assign_where
from sklearn.utils._mask import _get_mask
from sklearn.utils._testing import _convert_container, assert_allclose
def test_base_imputer_not_fit(data):
    imputer = NoFitIndicatorImputer(add_indicator=True)
    err_msg = 'Make sure to call _fit_indicator before _transform_indicator'
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(data).transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)