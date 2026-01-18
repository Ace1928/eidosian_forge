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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_fused_types_make_dataset(csr_container):
    iris = load_iris()
    X_32 = iris.data.astype(np.float32)
    y_32 = iris.target.astype(np.float32)
    X_csr_32 = csr_container(X_32)
    sample_weight_32 = np.arange(y_32.size, dtype=np.float32)
    X_64 = iris.data.astype(np.float64)
    y_64 = iris.target.astype(np.float64)
    X_csr_64 = csr_container(X_64)
    sample_weight_64 = np.arange(y_64.size, dtype=np.float64)
    dataset_32, _ = make_dataset(X_32, y_32, sample_weight_32)
    dataset_64, _ = make_dataset(X_64, y_64, sample_weight_64)
    xi_32, yi_32, _, _ = dataset_32._next_py()
    xi_64, yi_64, _, _ = dataset_64._next_py()
    xi_data_32, _, _ = xi_32
    xi_data_64, _, _ = xi_64
    assert xi_data_32.dtype == np.float32
    assert xi_data_64.dtype == np.float64
    assert_allclose(yi_64, yi_32, rtol=rtol)
    datasetcsr_32, _ = make_dataset(X_csr_32, y_32, sample_weight_32)
    datasetcsr_64, _ = make_dataset(X_csr_64, y_64, sample_weight_64)
    xicsr_32, yicsr_32, _, _ = datasetcsr_32._next_py()
    xicsr_64, yicsr_64, _, _ = datasetcsr_64._next_py()
    xicsr_data_32, _, _ = xicsr_32
    xicsr_data_64, _, _ = xicsr_64
    assert xicsr_data_32.dtype == np.float32
    assert xicsr_data_64.dtype == np.float64
    assert_allclose(xicsr_data_64, xicsr_data_32, rtol=rtol)
    assert_allclose(yicsr_64, yicsr_32, rtol=rtol)
    assert_array_equal(xi_data_32, xicsr_data_32)
    assert_array_equal(xi_data_64, xicsr_data_64)
    assert_array_equal(yi_32, yicsr_32)
    assert_array_equal(yi_64, yicsr_64)