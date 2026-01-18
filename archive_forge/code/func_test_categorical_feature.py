import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('n_bins', [15, 256])
def test_categorical_feature(n_bins):
    X = np.array([[4] * 500 + [1] * 3 + [10] * 4 + [0] * 4 + [13] + [7] * 5 + [np.nan] * 2], dtype=X_DTYPE).T
    known_categories = [np.unique(X[~np.isnan(X)])]
    bin_mapper = _BinMapper(n_bins=n_bins, is_categorical=np.array([True]), known_categories=known_categories).fit(X)
    assert bin_mapper.n_bins_non_missing_ == [6]
    assert_array_equal(bin_mapper.bin_thresholds_[0], [0, 1, 4, 7, 10, 13])
    X = np.array([[0, 1, 4, np.nan, 7, 10, 13]], dtype=X_DTYPE).T
    expected_trans = np.array([[0, 1, 2, n_bins - 1, 3, 4, 5]]).T
    assert_array_equal(bin_mapper.transform(X), expected_trans)
    X = np.array([[-4, -1, 100]], dtype=X_DTYPE).T
    expected_trans = np.array([[n_bins - 1, n_bins - 1, 6]]).T
    assert_array_equal(bin_mapper.transform(X), expected_trans)