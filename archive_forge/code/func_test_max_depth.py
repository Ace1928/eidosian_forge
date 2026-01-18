import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('max_depth', [1, 2, 3])
def test_max_depth(max_depth):
    rng = np.random.RandomState(seed=0)
    n_bins = 256
    n_samples = 1000
    X = rng.normal(size=(n_samples, 3))
    y = X[:, 0] - X[:, 1]
    mapper = _BinMapper(n_bins=n_bins)
    X = mapper.fit_transform(X)
    all_gradients = y.astype(G_H_DTYPE)
    all_hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    grower = TreeGrower(X, all_gradients, all_hessians, max_depth=max_depth)
    grower.grow()
    depth = max((leaf.depth for leaf in grower.finalized_leaves))
    assert depth == max_depth
    if max_depth == 1:
        assert_is_stump(grower)