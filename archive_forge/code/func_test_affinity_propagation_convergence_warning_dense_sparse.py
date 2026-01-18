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
@pytest.mark.parametrize('container', CSR_CONTAINERS + [np.array])
def test_affinity_propagation_convergence_warning_dense_sparse(container, global_dtype):
    """
    Check that having sparse or dense `centers` format should not
    influence the convergence.
    Non-regression test for gh-13334.
    """
    centers = container(np.zeros((1, 10)))
    rng = np.random.RandomState(42)
    X = rng.rand(40, 10).astype(global_dtype, copy=False)
    y = (4 * rng.rand(40)).astype(int)
    ap = AffinityPropagation(random_state=46)
    ap.fit(X, y)
    ap.cluster_centers_ = centers
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        assert_array_equal(ap.predict(X), np.zeros(X.shape[0], dtype=int))