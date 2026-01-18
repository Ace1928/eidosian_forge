import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
@if_safe_multiprocessing_with_blas
def test_fit_transform_parallel():
    alpha = 1
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)
    spca_lars = SparsePCA(n_components=3, method='lars', alpha=alpha, random_state=0)
    spca_lars.fit(Y)
    U1 = spca_lars.transform(Y)
    spca = SparsePCA(n_components=3, n_jobs=2, method='lars', alpha=alpha, random_state=0).fit(Y)
    U2 = spca.transform(Y)
    assert not np.all(spca_lars.components_ == 0)
    assert_array_almost_equal(U1, U2)