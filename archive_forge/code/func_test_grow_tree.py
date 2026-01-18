import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('n_bins, constant_hessian, stopping_param, shrinkage', [(11, True, 'min_gain_to_split', 0.5), (11, False, 'min_gain_to_split', 1.0), (11, True, 'max_leaf_nodes', 1.0), (11, False, 'max_leaf_nodes', 0.1), (42, True, 'max_leaf_nodes', 0.01), (42, False, 'max_leaf_nodes', 1.0), (256, True, 'min_gain_to_split', 1.0), (256, True, 'max_leaf_nodes', 0.1)])
def test_grow_tree(n_bins, constant_hessian, stopping_param, shrinkage):
    X_binned, all_gradients, all_hessians = _make_training_data(n_bins=n_bins, constant_hessian=constant_hessian)
    n_samples = X_binned.shape[0]
    if stopping_param == 'max_leaf_nodes':
        stopping_param = {'max_leaf_nodes': 3}
    else:
        stopping_param = {'min_gain_to_split': 0.01}
    grower = TreeGrower(X_binned, all_gradients, all_hessians, n_bins=n_bins, shrinkage=shrinkage, min_samples_leaf=1, **stopping_param)
    assert grower.root.left_child is None
    assert grower.root.right_child is None
    root_split = grower.root.split_info
    assert root_split.feature_idx == 0
    assert root_split.bin_idx == n_bins // 2
    assert len(grower.splittable_nodes) == 1
    left_node, right_node = grower.split_next()
    _check_children_consistency(grower.root, left_node, right_node)
    assert len(left_node.sample_indices) > 0.4 * n_samples
    assert len(left_node.sample_indices) < 0.6 * n_samples
    if grower.min_gain_to_split > 0:
        assert left_node.split_info.gain < grower.min_gain_to_split
        assert left_node in grower.finalized_leaves
    split_info = right_node.split_info
    assert split_info.gain > 1.0
    assert split_info.feature_idx == 1
    assert split_info.bin_idx == n_bins // 3
    assert right_node.left_child is None
    assert right_node.right_child is None
    assert len(grower.splittable_nodes) == 1
    right_left_node, right_right_node = grower.split_next()
    _check_children_consistency(right_node, right_left_node, right_right_node)
    assert len(right_left_node.sample_indices) > 0.1 * n_samples
    assert len(right_left_node.sample_indices) < 0.2 * n_samples
    assert len(right_right_node.sample_indices) > 0.2 * n_samples
    assert len(right_right_node.sample_indices) < 0.4 * n_samples
    assert not grower.splittable_nodes
    grower._apply_shrinkage()
    assert grower.root.left_child.value == approx(shrinkage)
    assert grower.root.right_child.left_child.value == approx(shrinkage)
    assert grower.root.right_child.right_child.value == approx(-shrinkage, rel=0.001)