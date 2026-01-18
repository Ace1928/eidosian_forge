import math
from itertools import product
import numpy as np
import pytest
from scipy.sparse import rand as sparse_rand
from sklearn import clone, datasets, manifold, neighbors, pipeline, preprocessing
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def create_sample_data(dtype, n_pts=25, add_noise=False):
    n_per_side = int(math.sqrt(n_pts))
    X = np.array(list(product(range(n_per_side), repeat=2))).astype(dtype, copy=False)
    if add_noise:
        rng = np.random.RandomState(0)
        noise = 0.1 * rng.randn(n_pts, 1).astype(dtype, copy=False)
        X = np.concatenate((X, noise), 1)
    return X