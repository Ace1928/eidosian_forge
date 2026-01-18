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
def assert_compatible_argkmin_results(neighbors_dists_a, neighbors_dists_b, neighbors_indices_a, neighbors_indices_b, rtol=1e-05, atol=1e-06):
    """Assert that argkmin results are valid up to rounding errors.

    This function asserts that the results of argkmin queries are valid up to:
    - rounding error tolerance on distance values;
    - permutations of indices for distances values that differ up to the
      expected precision level.

    Furthermore, the distances must be sorted.

    To be used for testing neighbors queries on float32 datasets: we accept
    neighbors rank swaps only if they are caused by small rounding errors on
    the distance computations.
    """
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert neighbors_dists_a.shape == neighbors_dists_b.shape == neighbors_indices_a.shape == neighbors_indices_b.shape, 'Arrays of results have incompatible shapes.'
    n_queries, _ = neighbors_dists_a.shape
    for query_idx in range(n_queries):
        dist_row_a = neighbors_dists_a[query_idx]
        dist_row_b = neighbors_dists_b[query_idx]
        indices_row_a = neighbors_indices_a[query_idx]
        indices_row_b = neighbors_indices_b[query_idx]
        assert is_sorted(dist_row_a), f"Distances aren't sorted on row {query_idx}"
        assert is_sorted(dist_row_b), f"Distances aren't sorted on row {query_idx}"
        assert_same_distances_for_common_neighbors(query_idx, dist_row_a, dist_row_b, indices_row_a, indices_row_b, rtol, atol)
        threshold = (1 - rtol) * np.maximum(np.max(dist_row_a), np.max(dist_row_b)) - atol
        assert_no_missing_neighbors(query_idx, dist_row_a, dist_row_b, indices_row_a, indices_row_b, threshold)