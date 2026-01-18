import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def generate_toy_data(n_components, n_samples, image_size, random_state=None):
    n_features = image_size[0] * image_size[1]
    rng = check_random_state(random_state)
    U = rng.randn(n_samples, n_components)
    V = rng.randn(n_components, n_features)
    centers = [(3, 3), (6, 7), (8, 1)]
    sz = [1, 2, 1]
    for k in range(n_components):
        img = np.zeros(image_size)
        xmin, xmax = (centers[k][0] - sz[k], centers[k][0] + sz[k])
        ymin, ymax = (centers[k][1] - sz[k], centers[k][1] + sz[k])
        img[xmin:xmax][:, ymin:ymax] = 1.0
        V[k, :] = img.ravel()
    Y = np.dot(U, V)
    Y += 0.1 * rng.randn(Y.shape[0], Y.shape[1])
    return (Y, U, V)