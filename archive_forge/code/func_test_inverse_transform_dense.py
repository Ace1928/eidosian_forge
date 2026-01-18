import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.fixes import CSC_CONTAINERS
def test_inverse_transform_dense():
    sel = StepSelector()
    Xinv_actual = sel.fit(X, y).inverse_transform(Xt)
    assert_array_equal(Xinv, Xinv_actual)
    assert np.int32 == sel.inverse_transform(Xt.astype(np.int32)).dtype
    assert np.float32 == sel.inverse_transform(Xt.astype(np.float32)).dtype
    names_inv_actual = sel.inverse_transform([feature_names_t])
    assert_array_equal(feature_names_inv, names_inv_actual.ravel())
    with pytest.raises(ValueError):
        sel.inverse_transform(np.array([[1], [2]]))