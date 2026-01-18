import numpy as np
import pytest
from sklearn.impute._base import _BaseImputer
from sklearn.impute._iterative import _assign_where
from sklearn.utils._mask import _get_mask
from sklearn.utils._testing import _convert_container, assert_allclose
def test_base_imputer_not_transform(data):
    imputer = NoTransformIndicatorImputer(add_indicator=True)
    err_msg = 'Call _fit_indicator and _transform_indicator in the imputer implementation'
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(data).transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)