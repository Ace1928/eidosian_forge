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
def test_affinity_propagation_fit_non_convergence(global_dtype):
    X = np.array([[0, 0], [1, 1], [-2, -2]], dtype=global_dtype)
    af = AffinityPropagation(preference=-10, max_iter=1, random_state=82)
    with pytest.warns(ConvergenceWarning):
        af.fit(X)
    assert_allclose(np.empty((0, 2)), af.cluster_centers_)
    assert_array_equal(np.array([-1, -1, -1]), af.labels_)