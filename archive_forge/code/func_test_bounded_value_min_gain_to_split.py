import re
import numpy as np
import pytest
from sklearn.ensemble import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import _convert_container
def test_bounded_value_min_gain_to_split():
    l2_regularization = 0
    min_hessian_to_split = 0
    min_samples_leaf = 1
    n_bins = n_samples = 5
    X_binned = np.arange(n_samples).reshape(-1, 1).astype(X_BINNED_DTYPE)
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_hessians = np.ones(n_samples, dtype=G_H_DTYPE)
    all_gradients = np.array([1, 1, 100, 1, 1], dtype=G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    sum_hessians = all_hessians.sum()
    hessians_are_constant = False
    builder = HistogramBuilder(X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads)
    n_bins_non_missing = np.array([n_bins - 1] * X_binned.shape[1], dtype=np.uint32)
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    monotonic_cst = np.array([MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8)
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    children_lower_bound, children_upper_bound = (-np.inf, np.inf)
    min_gain_to_split = 2000
    splitter = Splitter(X_binned, n_bins_non_missing, missing_values_bin_idx, has_missing_values, is_categorical, monotonic_cst, l2_regularization, min_hessian_to_split, min_samples_leaf, min_gain_to_split, hessians_are_constant)
    histograms = builder.compute_histograms_brute(sample_indices)
    current_lower_bound, current_upper_bound = (-np.inf, np.inf)
    value = compute_node_value(sum_gradients, sum_hessians, current_lower_bound, current_upper_bound, l2_regularization)
    assert value == pytest.approx(-104 / 5)
    split_info = splitter.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value, lower_bound=children_lower_bound, upper_bound=children_upper_bound)
    assert split_info.gain == -1
    current_lower_bound, current_upper_bound = (-10, np.inf)
    value = compute_node_value(sum_gradients, sum_hessians, current_lower_bound, current_upper_bound, l2_regularization)
    assert value == -10
    split_info = splitter.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value, lower_bound=children_lower_bound, upper_bound=children_upper_bound)
    assert split_info.gain > min_gain_to_split