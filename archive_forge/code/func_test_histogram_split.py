import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import skip_if_32bit
@pytest.mark.parametrize('n_bins', [3, 32, 256])
def test_histogram_split(n_bins):
    rng = np.random.RandomState(42)
    feature_idx = 0
    l2_regularization = 0
    min_hessian_to_split = 0.001
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    X_binned = np.asfortranarray(rng.randint(0, n_bins - 1, size=(int(10000.0), 1)), dtype=X_BINNED_DTYPE)
    binned_feature = X_binned.T[feature_idx]
    sample_indices = np.arange(binned_feature.shape[0], dtype=np.uint32)
    ordered_hessians = np.ones_like(binned_feature, dtype=G_H_DTYPE)
    all_hessians = ordered_hessians
    sum_hessians = all_hessians.sum()
    hessians_are_constant = False
    for true_bin in range(1, n_bins - 2):
        for sign in [-1, 1]:
            ordered_gradients = np.full_like(binned_feature, sign, dtype=G_H_DTYPE)
            ordered_gradients[binned_feature <= true_bin] *= -1
            all_gradients = ordered_gradients
            sum_gradients = all_gradients.sum()
            builder = HistogramBuilder(X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads)
            n_bins_non_missing = np.array([n_bins - 1] * X_binned.shape[1], dtype=np.uint32)
            has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
            monotonic_cst = np.array([MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8)
            is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
            missing_values_bin_idx = n_bins - 1
            splitter = Splitter(X_binned, n_bins_non_missing, missing_values_bin_idx, has_missing_values, is_categorical, monotonic_cst, l2_regularization, min_hessian_to_split, min_samples_leaf, min_gain_to_split, hessians_are_constant)
            histograms = builder.compute_histograms_brute(sample_indices)
            value = compute_node_value(sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization)
            split_info = splitter.find_node_split(sample_indices.shape[0], histograms, sum_gradients, sum_hessians, value)
            assert split_info.bin_idx == true_bin
            assert split_info.gain >= 0
            assert split_info.feature_idx == feature_idx
            assert split_info.n_samples_left + split_info.n_samples_right == sample_indices.shape[0]
            assert split_info.n_samples_left == split_info.sum_hessian_left