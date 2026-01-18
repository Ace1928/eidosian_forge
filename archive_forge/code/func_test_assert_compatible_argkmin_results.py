import itertools
import re
import warnings
from functools import partial
import numpy as np
import pytest
import threadpoolctl
from scipy.spatial.distance import cdist
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics._pairwise_distances_reduction import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_assert_compatible_argkmin_results():
    atol = 1e-07
    rtol = 0.0
    tols = dict(atol=atol, rtol=rtol)
    eps = atol / 3
    _1m = 1.0 - eps
    _1p = 1.0 + eps
    _6_1m = 6.1 - eps
    _6_1p = 6.1 + eps
    ref_dist = np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p], [_1m, _1m, 1, _1p, _1p]])
    ref_indices = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    assert_compatible_argkmin_results(ref_dist, ref_dist, ref_indices, ref_indices, rtol)
    assert_compatible_argkmin_results(np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]), np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]), np.array([[1, 2, 3, 4, 5]]), np.array([[1, 2, 5, 4, 3]]), **tols)
    assert_compatible_argkmin_results(np.array([[1.2, 2.5, 3.0, 6.1, _6_1p]]), np.array([[1.2, 2.5, 3.0, _6_1m, 6.1]]), np.array([[1, 2, 3, 4, 5]]), np.array([[1, 2, 3, 6, 7]]), **tols)
    assert_compatible_argkmin_results(np.array([[_1m, 1, _1p, _1p, _1p]]), np.array([[1, 1, 1, 1, _1p]]), np.array([[7, 6, 8, 10, 9]]), np.array([[6, 9, 7, 8, 10]]), **tols)
    assert_compatible_argkmin_results(np.array([[_1m, 1, _1p, _1p, _1p]]), np.array([[_1m, 1, 1, 1, _1p]]), np.array([[34, 30, 8, 12, 24]]), np.array([[42, 1, 21, 13, 3]]), **tols)
    msg = re.escape('Query vector with index 0 lead to different distances for common neighbor with index 1: dist_a=1.2 vs dist_b=2.5')
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]), np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]), np.array([[1, 2, 3, 4, 5]]), np.array([[2, 1, 3, 4, 5]]), **tols)
    msg = re.escape('neighbors in b missing from a: [12]\nneighbors in a missing from b: [1]')
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]), np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]), np.array([[1, 2, 3, 4, 5]]), np.array([[12, 2, 4, 11, 3]]), **tols)
    msg = re.escape('neighbors in b missing from a: []\nneighbors in a missing from b: [3]')
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(np.array([[_1m, 1.0, _6_1m, 6.1, _6_1p]]), np.array([[1.0, 1.0, _6_1m, 6.1, 7]]), np.array([[1, 2, 3, 4, 5]]), np.array([[2, 1, 4, 5, 12]]), **tols)
    msg = re.escape('neighbors in b missing from a: [5]\nneighbors in a missing from b: []')
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(np.array([[_1m, 1.0, _6_1m, 6.1, 7]]), np.array([[1.0, 1.0, _6_1m, 6.1, _6_1p]]), np.array([[1, 2, 3, 4, 12]]), np.array([[2, 1, 5, 3, 4]]), **tols)
    msg = "Distances aren't sorted on row 0"
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]), np.array([[2.5, 1.2, _6_1m, 6.1, _6_1p]]), np.array([[1, 2, 3, 4, 5]]), np.array([[2, 1, 4, 5, 3]]), **tols)