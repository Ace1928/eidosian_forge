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
def test_correct_clusters(global_dtype):
    X = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=global_dtype)
    afp = AffinityPropagation(preference=1, affinity='precomputed', random_state=0).fit(X)
    expected = np.array([0, 1, 1, 2])
    assert_array_equal(afp.labels_, expected)