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
def test_isomap_fitted_attributes_dtype(global_dtype):
    """Check that the fitted attributes are stored accordingly to the
    data type of X."""
    iso = manifold.Isomap(n_neighbors=2)
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=global_dtype)
    iso.fit(X)
    assert iso.dist_matrix_.dtype == global_dtype
    assert iso.embedding_.dtype == global_dtype