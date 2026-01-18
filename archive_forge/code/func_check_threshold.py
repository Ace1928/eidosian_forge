import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def check_threshold(birch_instance, threshold):
    """Use the leaf linked list for traversal"""
    current_leaf = birch_instance.dummy_leaf_.next_leaf_
    while current_leaf:
        subclusters = current_leaf.subclusters_
        for sc in subclusters:
            assert threshold >= sc.radius
        current_leaf = current_leaf.next_leaf_