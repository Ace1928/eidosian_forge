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
def test_isomap_raise_error_when_neighbor_and_radius_both_set():
    X, _ = datasets.load_digits(return_X_y=True)
    isomap = manifold.Isomap(n_neighbors=3, radius=5.5)
    msg = 'Both n_neighbors and radius are provided'
    with pytest.raises(ValueError, match=msg):
        isomap.fit_transform(X)