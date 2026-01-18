import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('max_bins', [5, 10, 42])
def test_bin_mapper_random_data(max_bins):
    n_samples, n_features = DATA.shape
    expected_count_per_bin = n_samples // max_bins
    tol = int(0.05 * expected_count_per_bin)
    n_bins = max_bins + 1
    mapper = _BinMapper(n_bins=n_bins, random_state=42).fit(DATA)
    binned = mapper.transform(DATA)
    assert binned.shape == (n_samples, n_features)
    assert binned.dtype == np.uint8
    assert_array_equal(binned.min(axis=0), np.array([0, 0]))
    assert_array_equal(binned.max(axis=0), np.array([max_bins - 1, max_bins - 1]))
    assert len(mapper.bin_thresholds_) == n_features
    for bin_thresholds_feature in mapper.bin_thresholds_:
        assert bin_thresholds_feature.shape == (max_bins - 1,)
        assert bin_thresholds_feature.dtype == DATA.dtype
    assert np.all(mapper.n_bins_non_missing_ == max_bins)
    for feature_idx in range(n_features):
        for bin_idx in range(max_bins):
            count = (binned[:, feature_idx] == bin_idx).sum()
            assert abs(count - expected_count_per_bin) < tol