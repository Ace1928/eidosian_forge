import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_mini_batch_correct_shapes():
    rng = np.random.RandomState(0)
    X = rng.randn(12, 10)
    pca = MiniBatchSparsePCA(n_components=8, max_iter=1, random_state=rng)
    U = pca.fit_transform(X)
    assert pca.components_.shape == (8, 10)
    assert U.shape == (12, 8)
    pca = MiniBatchSparsePCA(n_components=13, max_iter=1, random_state=rng)
    U = pca.fit_transform(X)
    assert pca.components_.shape == (13, 10)
    assert U.shape == (12, 13)