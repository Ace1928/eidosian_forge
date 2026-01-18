import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('n_bins, n_bins_non_missing, X_trans_expected', [(256, [4, 2, 2], [[0, 0, 0], [255, 255, 0], [1, 0, 0], [255, 1, 1], [2, 1, 1], [3, 0, 0]]), (3, [2, 2, 2], [[0, 0, 0], [2, 2, 0], [0, 0, 0], [2, 1, 1], [1, 1, 1], [1, 0, 0]])])
def test_missing_values_support(n_bins, n_bins_non_missing, X_trans_expected):
    X = [[1, 1, 0], [np.nan, np.nan, 0], [2, 1, 0], [np.nan, 2, 1], [3, 2, 1], [4, 1, 0]]
    X = np.array(X)
    mapper = _BinMapper(n_bins=n_bins)
    mapper.fit(X)
    assert_array_equal(mapper.n_bins_non_missing_, n_bins_non_missing)
    for feature_idx in range(X.shape[1]):
        assert len(mapper.bin_thresholds_[feature_idx]) == n_bins_non_missing[feature_idx] - 1
    assert mapper.missing_values_bin_idx_ == n_bins - 1
    X_trans = mapper.transform(X)
    assert_array_equal(X_trans, X_trans_expected)