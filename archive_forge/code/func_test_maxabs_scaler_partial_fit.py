import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_maxabs_scaler_partial_fit(csr_container):
    X = X_2d[:100, :]
    n = X.shape[0]
    for chunk_size in [1, 2, 50, n, n + 42]:
        scaler_batch = MaxAbsScaler().fit(X)
        scaler_incr = MaxAbsScaler()
        scaler_incr_csr = MaxAbsScaler()
        scaler_incr_csc = MaxAbsScaler()
        for batch in gen_batches(n, chunk_size):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            X_csr = csr_container(X[batch])
            scaler_incr_csr = scaler_incr_csr.partial_fit(X_csr)
            X_csc = csr_container(X[batch])
            scaler_incr_csc = scaler_incr_csc.partial_fit(X_csc)
        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr.max_abs_)
        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr_csr.max_abs_)
        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr_csc.max_abs_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        assert scaler_batch.n_samples_seen_ == scaler_incr_csr.n_samples_seen_
        assert scaler_batch.n_samples_seen_ == scaler_incr_csc.n_samples_seen_
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr_csr.scale_)
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr_csc.scale_)
        assert_array_almost_equal(scaler_batch.transform(X), scaler_incr.transform(X))
        batch0 = slice(0, chunk_size)
        scaler_batch = MaxAbsScaler().fit(X[batch0])
        scaler_incr = MaxAbsScaler().partial_fit(X[batch0])
        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr.max_abs_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_almost_equal(scaler_batch.transform(X), scaler_incr.transform(X))
        scaler_batch = MaxAbsScaler().fit(X)
        scaler_incr = MaxAbsScaler()
        for i, batch in enumerate(gen_batches(n, chunk_size)):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            assert_correct_incr(i, batch_start=batch.start, batch_stop=batch.stop, n=n, chunk_size=chunk_size, n_samples_seen=scaler_incr.n_samples_seen_)