import warnings
import numpy as np
import pytest
from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, mean_shift
from sklearn.datasets import make_blobs
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
def test_mean_shift_zero_bandwidth(global_dtype):
    X = np.array([1, 1, 1, 2, 2, 2, 3, 3], dtype=global_dtype).reshape(-1, 1)
    bandwidth = estimate_bandwidth(X)
    assert bandwidth == 0
    assert get_bin_seeds(X, bin_size=bandwidth) is X
    ms_binning = MeanShift(bin_seeding=True, bandwidth=None).fit(X)
    ms_nobinning = MeanShift(bin_seeding=False).fit(X)
    expected_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    assert v_measure_score(ms_binning.labels_, expected_labels) == pytest.approx(1)
    assert v_measure_score(ms_nobinning.labels_, expected_labels) == pytest.approx(1)
    assert_allclose(ms_binning.cluster_centers_, ms_nobinning.cluster_centers_)