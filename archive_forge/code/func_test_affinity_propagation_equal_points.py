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
def test_affinity_propagation_equal_points():
    """Make sure we do not assign multiple clusters to equal points.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/20043
    """
    X = np.zeros((8, 1))
    af = AffinityPropagation(affinity='euclidean', damping=0.5, random_state=42).fit(X)
    assert np.all(af.labels_ == 0)