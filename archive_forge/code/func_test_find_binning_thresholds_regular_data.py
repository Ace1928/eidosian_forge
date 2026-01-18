import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def test_find_binning_thresholds_regular_data():
    data = np.linspace(0, 10, 1001)
    bin_thresholds = _find_binning_thresholds(data, max_bins=10)
    assert_allclose(bin_thresholds, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    bin_thresholds = _find_binning_thresholds(data, max_bins=5)
    assert_allclose(bin_thresholds, [2, 4, 6, 8])