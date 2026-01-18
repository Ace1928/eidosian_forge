import warnings
import numpy as np
import pytest
from scipy import linalg, sparse
from sklearn.datasets import load_iris, make_regression, make_sparse_uncorrelated
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import (
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
@pytest.mark.parametrize('use_sw', [True, False])
def test_inplace_data_preprocessing(sparse_container, use_sw, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    original_X_data = rng.randn(10, 12)
    original_y_data = rng.randn(10, 2)
    orginal_sw_data = rng.rand(10)
    if sparse_container is not None:
        X = sparse_container(original_X_data)
    else:
        X = original_X_data.copy()
    y = original_y_data.copy()
    if use_sw:
        sample_weight = orginal_sw_data.copy()
    else:
        sample_weight = None
    reg = LinearRegression()
    reg.fit(X, y, sample_weight=sample_weight)
    if sparse_container is not None:
        assert_allclose(X.toarray(), original_X_data)
    else:
        assert_allclose(X, original_X_data)
    assert_allclose(y, original_y_data)
    if use_sw:
        assert_allclose(sample_weight, orginal_sw_data)
    reg = LinearRegression(copy_X=False)
    reg.fit(X, y, sample_weight=sample_weight)
    if sparse_container is not None:
        assert_allclose(X.toarray(), original_X_data)
    else:
        assert np.linalg.norm(X - original_X_data) > 0.42
    assert_allclose(y, original_y_data)
    if use_sw:
        assert_allclose(sample_weight, orginal_sw_data)