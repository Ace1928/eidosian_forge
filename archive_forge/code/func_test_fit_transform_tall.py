import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_fit_transform_tall():
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 65, (8, 8), random_state=rng)
    spca_lars = SparsePCA(n_components=3, method='lars', random_state=rng)
    U1 = spca_lars.fit_transform(Y)
    spca_lasso = SparsePCA(n_components=3, method='cd', random_state=rng)
    U2 = spca_lasso.fit(Y).transform(Y)
    assert_array_almost_equal(U1, U2)