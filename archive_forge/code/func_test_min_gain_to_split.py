import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import skip_if_32bit
def test_min_gain_to_split():
    rng = np.random.RandomState(42)
    l2_regularization = 0
    min_hessian_to_split = 0
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    n_bins = 255
    n_samples = 100
    X_binned = np.asfortranarray(rng.randint(0, n_bins, size=(n_samples, 1)), dtype=X_BINNED_DTYPE)
    binned_feature = X_binned[:, 0]
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_hessians = np.ones_like(binned_feature, dtype=G_H_DTYPE)
    all_gradients = np.ones_like(binned_feature, dtype=G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    sum_hessians = all_hessians.sum()
    hessians_are_constant = False
    builder = HistogramBuilder(X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads)
    n_bins_non_missing = np.array([n_bins - 1] * X_binned.shape[1], dtype=np.uint32)
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    monotonic_cst = np.array([MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8)
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    splitter = Splitter(X_binned, n_bins_non_missing, missing_values_bin_idx, has_missing_values, is_categorical, monotonic_cst, l2_regularization, min_hessian_to_split, min_samples_leaf, min_gain_to_split, hessians_are_constant)
    histograms = builder.compute_histograms_brute(sample_indices)
    value = compute_node_value(sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization)
    split_info = splitter.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value)
    assert split_info.gain == -1