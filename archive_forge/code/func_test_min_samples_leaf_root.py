import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('n_samples, min_samples_leaf', [(99, 50), (100, 50)])
def test_min_samples_leaf_root(n_samples, min_samples_leaf):
    rng = np.random.RandomState(seed=0)
    n_bins = 256
    X = rng.normal(size=(n_samples, 3))
    y = X[:, 0] - X[:, 1]
    mapper = _BinMapper(n_bins=n_bins)
    X = mapper.fit_transform(X)
    all_gradients = y.astype(G_H_DTYPE)
    all_hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    grower = TreeGrower(X, all_gradients, all_hessians, n_bins=n_bins, shrinkage=1.0, min_samples_leaf=min_samples_leaf, max_leaf_nodes=n_samples)
    grower.grow()
    if n_samples >= min_samples_leaf * 2:
        assert len(grower.finalized_leaves) >= 2
    else:
        assert len(grower.finalized_leaves) == 1