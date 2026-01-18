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
def test_affinity_propagation_precomputed():
    """Check equality of precomputed affinity matrix to internally computed affinity
    matrix.
    """
    S = -euclidean_distances(X, squared=True)
    preference = np.median(S) * 10
    af = AffinityPropagation(preference=preference, affinity='precomputed', random_state=28)
    labels_precomputed = af.fit(S).labels_
    af = AffinityPropagation(preference=preference, verbose=True, random_state=37)
    labels = af.fit(X).labels_
    assert_array_equal(labels, labels_precomputed)
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    assert np.unique(labels).size == n_clusters_
    assert n_clusters == n_clusters_