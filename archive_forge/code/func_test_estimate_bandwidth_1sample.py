import warnings
import numpy as np
import pytest
from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, mean_shift
from sklearn.datasets import make_blobs
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
def test_estimate_bandwidth_1sample(global_dtype):
    bandwidth = estimate_bandwidth(X.astype(global_dtype, copy=False), n_samples=1, quantile=0.3)
    assert bandwidth.dtype == X.dtype
    assert bandwidth == pytest.approx(0.0, abs=1e-05)