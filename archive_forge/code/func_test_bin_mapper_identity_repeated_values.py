import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('max_bins, n_distinct, multiplier', [(5, 5, 1), (5, 5, 3), (255, 12, 42)])
def test_bin_mapper_identity_repeated_values(max_bins, n_distinct, multiplier):
    data = np.array(list(range(n_distinct)) * multiplier).reshape(-1, 1)
    n_bins = max_bins + 1
    binned = _BinMapper(n_bins=n_bins).fit_transform(data)
    assert_array_equal(data, binned)