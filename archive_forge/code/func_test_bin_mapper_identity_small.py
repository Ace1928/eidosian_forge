import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('max_bins, scale, offset', [(3, 2, -1), (42, 1, 0), (255, 0.3, 42)])
def test_bin_mapper_identity_small(max_bins, scale, offset):
    data = np.arange(max_bins).reshape(-1, 1) * scale + offset
    n_bins = max_bins + 1
    binned = _BinMapper(n_bins=n_bins).fit_transform(data)
    assert_array_equal(binned, np.arange(max_bins).reshape(-1, 1))