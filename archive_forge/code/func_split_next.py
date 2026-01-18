import numbers
from heapq import heappop, heappush
from timeit import default_timer as time
import numpy as np
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from ._bitset import set_raw_bitset_from_binned_bitset
from .common import (
from .histogram import HistogramBuilder
from .predictor import TreePredictor
from .splitting import Splitter
from .utils import sum_parallel
def split_next(self):
    """Split the node with highest potential gain.

        Returns
        -------
        left : TreeNode
            The resulting left child.
        right : TreeNode
            The resulting right child.
        """
    node = heappop(self.splittable_nodes)
    tic = time()
    sample_indices_left, sample_indices_right, right_child_pos = self.splitter.split_indices(node.split_info, node.sample_indices)
    self.total_apply_split_time += time() - tic
    depth = node.depth + 1
    n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes)
    n_leaf_nodes += 2
    left_child_node = TreeNode(depth, sample_indices_left, node.split_info.sum_gradient_left, node.split_info.sum_hessian_left, value=node.split_info.value_left)
    right_child_node = TreeNode(depth, sample_indices_right, node.split_info.sum_gradient_right, node.split_info.sum_hessian_right, value=node.split_info.value_right)
    node.right_child = right_child_node
    node.left_child = left_child_node
    left_child_node.partition_start = node.partition_start
    left_child_node.partition_stop = node.partition_start + right_child_pos
    right_child_node.partition_start = left_child_node.partition_stop
    right_child_node.partition_stop = node.partition_stop
    if self.interaction_cst is not None:
        left_child_node.allowed_features, left_child_node.interaction_cst_indices = self._compute_interactions(node)
        right_child_node.interaction_cst_indices = left_child_node.interaction_cst_indices
        right_child_node.allowed_features = left_child_node.allowed_features
    if not self.has_missing_values[node.split_info.feature_idx]:
        node.split_info.missing_go_to_left = left_child_node.n_samples > right_child_node.n_samples
    self.n_nodes += 2
    self.n_categorical_splits += node.split_info.is_categorical
    if self.max_leaf_nodes is not None and n_leaf_nodes == self.max_leaf_nodes:
        self._finalize_leaf(left_child_node)
        self._finalize_leaf(right_child_node)
        self._finalize_splittable_nodes()
        return (left_child_node, right_child_node)
    if self.max_depth is not None and depth == self.max_depth:
        self._finalize_leaf(left_child_node)
        self._finalize_leaf(right_child_node)
        return (left_child_node, right_child_node)
    if left_child_node.n_samples < self.min_samples_leaf * 2:
        self._finalize_leaf(left_child_node)
    if right_child_node.n_samples < self.min_samples_leaf * 2:
        self._finalize_leaf(right_child_node)
    if self.with_monotonic_cst:
        if self.monotonic_cst[node.split_info.feature_idx] == MonotonicConstraint.NO_CST:
            lower_left = lower_right = node.children_lower_bound
            upper_left = upper_right = node.children_upper_bound
        else:
            mid = (left_child_node.value + right_child_node.value) / 2
            if self.monotonic_cst[node.split_info.feature_idx] == MonotonicConstraint.POS:
                lower_left, upper_left = (node.children_lower_bound, mid)
                lower_right, upper_right = (mid, node.children_upper_bound)
            else:
                lower_left, upper_left = (mid, node.children_upper_bound)
                lower_right, upper_right = (node.children_lower_bound, mid)
        left_child_node.set_children_bounds(lower_left, upper_left)
        right_child_node.set_children_bounds(lower_right, upper_right)
    should_split_left = not left_child_node.is_leaf
    should_split_right = not right_child_node.is_leaf
    if should_split_left or should_split_right:
        n_samples_left = left_child_node.sample_indices.shape[0]
        n_samples_right = right_child_node.sample_indices.shape[0]
        if n_samples_left < n_samples_right:
            smallest_child = left_child_node
            largest_child = right_child_node
        else:
            smallest_child = right_child_node
            largest_child = left_child_node
        tic = time()
        smallest_child.histograms = self.histogram_builder.compute_histograms_brute(smallest_child.sample_indices, smallest_child.allowed_features)
        largest_child.histograms = self.histogram_builder.compute_histograms_subtraction(node.histograms, smallest_child.histograms, smallest_child.allowed_features)
        node.histograms = None
        self.total_compute_hist_time += time() - tic
        tic = time()
        if should_split_left:
            self._compute_best_split_and_push(left_child_node)
        if should_split_right:
            self._compute_best_split_and_push(right_child_node)
        self.total_find_split_time += time() - tic
        for child in (left_child_node, right_child_node):
            if child.is_leaf:
                del child.histograms
    del node.histograms
    return (left_child_node, right_child_node)