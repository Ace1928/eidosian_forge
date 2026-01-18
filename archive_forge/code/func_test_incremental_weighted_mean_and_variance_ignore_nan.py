import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.special import expit
from sklearn.datasets import make_low_rank_matrix, make_sparse_spd_matrix
from sklearn.utils import gen_batches
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._testing import (
from sklearn.utils.extmath import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_incremental_weighted_mean_and_variance_ignore_nan(dtype):
    old_means = np.array([535.0, 535.0, 535.0, 535.0])
    old_variances = np.array([4225.0, 4225.0, 4225.0, 4225.0])
    old_weight_sum = np.array([2, 2, 2, 2], dtype=np.int32)
    sample_weights_X = np.ones(3)
    sample_weights_X_nan = np.ones(4)
    X = np.array([[170, 170, 170, 170], [430, 430, 430, 430], [300, 300, 300, 300]]).astype(dtype)
    X_nan = np.array([[170, np.nan, 170, 170], [np.nan, 170, 430, 430], [430, 430, np.nan, 300], [300, 300, 300, np.nan]]).astype(dtype)
    X_means, X_variances, X_count = _incremental_mean_and_var(X, old_means, old_variances, old_weight_sum, sample_weight=sample_weights_X)
    X_nan_means, X_nan_variances, X_nan_count = _incremental_mean_and_var(X_nan, old_means, old_variances, old_weight_sum, sample_weight=sample_weights_X_nan)
    assert_allclose(X_nan_means, X_means)
    assert_allclose(X_nan_variances, X_variances)
    assert_allclose(X_nan_count, X_count)