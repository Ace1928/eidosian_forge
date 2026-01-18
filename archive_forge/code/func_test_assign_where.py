import numpy as np
import pytest
from sklearn.impute._base import _BaseImputer
from sklearn.impute._iterative import _assign_where
from sklearn.utils._mask import _get_mask
from sklearn.utils._testing import _convert_container, assert_allclose
@pytest.mark.parametrize('X1_type', ['array', 'dataframe'])
def test_assign_where(X1_type):
    """Check the behaviour of the private helpers `_assign_where`."""
    rng = np.random.RandomState(0)
    n_samples, n_features = (10, 5)
    X1 = _convert_container(rng.randn(n_samples, n_features), constructor_name=X1_type)
    X2 = rng.randn(n_samples, n_features)
    mask = rng.randint(0, 2, size=(n_samples, n_features)).astype(bool)
    _assign_where(X1, X2, mask)
    if X1_type == 'dataframe':
        X1 = X1.to_numpy()
    assert_allclose(X1[mask], X2[mask])