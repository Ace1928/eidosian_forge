import warnings
import numpy as np
import pytest
from sklearn.cluster import AffinityPropagation, affinity_propagation
from sklearn.cluster._affinity_propagation import _equal_similarities_and_preferences
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_equal_similarities_and_preferences(global_dtype):
    X = np.array([[0, 0], [1, 1], [-2, -2]], dtype=global_dtype)
    S = -euclidean_distances(X, squared=True)
    assert not _equal_similarities_and_preferences(S, np.array(0))
    assert not _equal_similarities_and_preferences(S, np.array([0, 0]))
    assert not _equal_similarities_and_preferences(S, np.array([0, 1]))
    X = np.array([[0, 0], [1, 1]], dtype=global_dtype)
    S = -euclidean_distances(X, squared=True)
    assert not _equal_similarities_and_preferences(S, np.array([0, 1]))
    assert _equal_similarities_and_preferences(S, np.array([0, 0]))
    assert _equal_similarities_and_preferences(S, np.array(0))