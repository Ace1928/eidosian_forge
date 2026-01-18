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
def test_isomap_dtype_equivalence():
    """Check the equivalence of the results with 32 and 64 bits input."""
    iso_32 = manifold.Isomap(n_neighbors=2)
    X_32 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    iso_32.fit(X_32)
    iso_64 = manifold.Isomap(n_neighbors=2)
    X_64 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    iso_64.fit(X_64)
    assert_allclose(iso_32.dist_matrix_, iso_64.dist_matrix_)