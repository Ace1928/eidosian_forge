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
def assert_same_distances_for_common_neighbors(query_idx, dist_row_a, dist_row_b, indices_row_a, indices_row_b, rtol, atol):
    """Check that the distances of common neighbors are equal up to tolerance.

    This does not check if there are missing neighbors in either result set.
    Missingness is handled by assert_no_missing_neighbors.
    """
    indices_to_dist_a = dict(zip(indices_row_a, dist_row_a))
    indices_to_dist_b = dict(zip(indices_row_b, dist_row_b))
    common_indices = set(indices_row_a).intersection(set(indices_row_b))
    for idx in common_indices:
        dist_a = indices_to_dist_a[idx]
        dist_b = indices_to_dist_b[idx]
        try:
            assert_allclose(dist_a, dist_b, rtol=rtol, atol=atol)
        except AssertionError as e:
            raise AssertionError(f'Query vector with index {query_idx} lead to different distances for common neighbor with index {idx}: dist_a={dist_a} vs dist_b={dist_b} (with atol={atol} and rtol={rtol})') from e