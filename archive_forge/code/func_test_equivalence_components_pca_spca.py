import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_equivalence_components_pca_spca(global_random_seed):
    """Check the equivalence of the components found by PCA and SparsePCA.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/23932
    """
    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(50, 4)
    n_components = 2
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=0).fit(X)
    spca = SparsePCA(n_components=n_components, method='lars', ridge_alpha=0, alpha=0, random_state=0).fit(X)
    assert_allclose(pca.components_, spca.components_)