import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def test_categorical_feature_negative_missing():
    """Make sure bin mapper treats negative categories as missing values."""
    X = np.array([[4] * 500 + [1] * 3 + [5] * 10 + [-1] * 3 + [np.nan] * 4], dtype=X_DTYPE).T
    bin_mapper = _BinMapper(n_bins=4, is_categorical=np.array([True]), known_categories=[np.array([1, 4, 5], dtype=X_DTYPE)]).fit(X)
    assert bin_mapper.n_bins_non_missing_ == [3]
    X = np.array([[-1, 1, 3, 5, np.nan]], dtype=X_DTYPE).T
    assert bin_mapper.missing_values_bin_idx_ == 3
    expected_trans = np.array([[3, 0, 1, 2, 3]]).T
    assert_array_equal(bin_mapper.transform(X), expected_trans)