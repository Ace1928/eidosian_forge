import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def test_grow_tree_categories():
    X_binned = np.array([[0, 1] * 11 + [1]], dtype=X_BINNED_DTYPE).T
    X_binned = np.asfortranarray(X_binned)
    all_gradients = np.array([10, 1] * 11 + [1], dtype=G_H_DTYPE)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    is_categorical = np.ones(1, dtype=np.uint8)
    grower = TreeGrower(X_binned, all_gradients, all_hessians, n_bins=4, shrinkage=1.0, min_samples_leaf=1, is_categorical=is_categorical, n_threads=n_threads)
    grower.grow()
    assert grower.n_nodes == 3
    categories = [np.array([4, 9], dtype=X_DTYPE)]
    predictor = grower.make_predictor(binning_thresholds=categories)
    root = predictor.nodes[0]
    assert root['count'] == 23
    assert root['depth'] == 0
    assert root['is_categorical']
    left, right = (predictor.nodes[root['left']], predictor.nodes[root['right']])
    assert left['count'] >= right['count']
    expected_binned_cat_bitset = [2 ** 1] + [0] * 7
    binned_cat_bitset = predictor.binned_left_cat_bitsets
    assert_array_equal(binned_cat_bitset[0], expected_binned_cat_bitset)
    expected_raw_cat_bitsets = [2 ** 9] + [0] * 7
    raw_cat_bitsets = predictor.raw_left_cat_bitsets
    assert_array_equal(raw_cat_bitsets[0], expected_raw_cat_bitsets)
    assert root['missing_go_to_left']
    prediction_binned = predictor.predict_binned(np.asarray([[6]]).astype(X_BINNED_DTYPE), missing_values_bin_idx=6, n_threads=n_threads)
    assert_allclose(prediction_binned, [-1])
    known_cat_bitsets = np.zeros((1, 8), dtype=np.uint32)
    f_idx_map = np.array([0], dtype=np.uint32)
    prediction = predictor.predict(np.array([[np.nan]]), known_cat_bitsets, f_idx_map, n_threads)
    assert_allclose(prediction, [-1])