import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('n_samples, max_bins', [(5, 5), (5, 10), (5, 11), (42, 255)])
def test_bin_mapper_small_random_data(n_samples, max_bins):
    data = np.random.RandomState(42).normal(size=n_samples).reshape(-1, 1)
    assert len(np.unique(data)) == n_samples
    n_bins = max_bins + 1
    mapper = _BinMapper(n_bins=n_bins, random_state=42)
    binned = mapper.fit_transform(data)
    assert binned.shape == data.shape
    assert binned.dtype == np.uint8
    assert_array_equal(binned.ravel()[np.argsort(data.ravel())], np.arange(n_samples))