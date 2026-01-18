import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('n_bins', [10, 100, 256])
@pytest.mark.parametrize('diff', [-5, 0, 5])
def test_n_bins_non_missing(n_bins, diff):
    n_unique_values = n_bins + diff
    X = list(range(n_unique_values)) * 2
    X = np.array(X).reshape(-1, 1)
    mapper = _BinMapper(n_bins=n_bins).fit(X)
    assert np.all(mapper.n_bins_non_missing_ == min(n_bins - 1, n_unique_values))