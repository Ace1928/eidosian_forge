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
def test_incremental_variance_ddof():
    rng = np.random.RandomState(1999)
    X = rng.randn(50, 10)
    n_samples, n_features = X.shape
    for batch_size in [11, 20, 37]:
        steps = np.arange(0, X.shape[0], batch_size)
        if steps[-1] != X.shape[0]:
            steps = np.hstack([steps, n_samples])
        for i, j in zip(steps[:-1], steps[1:]):
            batch = X[i:j, :]
            if i == 0:
                incremental_means = batch.mean(axis=0)
                incremental_variances = batch.var(axis=0)
                incremental_count = batch.shape[0]
                sample_count = np.full(batch.shape[1], batch.shape[0], dtype=np.int32)
            else:
                result = _incremental_mean_and_var(batch, incremental_means, incremental_variances, sample_count)
                incremental_means, incremental_variances, incremental_count = result
                sample_count += batch.shape[0]
            calculated_means = np.mean(X[:j], axis=0)
            calculated_variances = np.var(X[:j], axis=0)
            assert_almost_equal(incremental_means, calculated_means, 6)
            assert_almost_equal(incremental_variances, calculated_variances, 6)
            assert_array_equal(incremental_count, sample_count)