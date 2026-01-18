import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import skip_if_32bit
@skip_if_32bit
@pytest.mark.parametrize('constant_hessian', [True, False])
def test_gradient_and_hessian_sanity(constant_hessian):
    rng = np.random.RandomState(42)
    n_bins = 10
    n_features = 20
    n_samples = 500
    l2_regularization = 0.0
    min_hessian_to_split = 0.001
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    X_binned = rng.randint(0, n_bins, size=(n_samples, n_features), dtype=X_BINNED_DTYPE)
    X_binned = np.asfortranarray(X_binned)
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_gradients = rng.randn(n_samples).astype(G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    if constant_hessian:
        all_hessians = np.ones(1, dtype=G_H_DTYPE)
        sum_hessians = 1 * n_samples
    else:
        all_hessians = rng.lognormal(size=n_samples).astype(G_H_DTYPE)
        sum_hessians = all_hessians.sum()
    builder = HistogramBuilder(X_binned, n_bins, all_gradients, all_hessians, constant_hessian, n_threads)
    n_bins_non_missing = np.array([n_bins - 1] * X_binned.shape[1], dtype=np.uint32)
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    monotonic_cst = np.array([MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8)
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    splitter = Splitter(X_binned, n_bins_non_missing, missing_values_bin_idx, has_missing_values, is_categorical, monotonic_cst, l2_regularization, min_hessian_to_split, min_samples_leaf, min_gain_to_split, constant_hessian)
    hists_parent = builder.compute_histograms_brute(sample_indices)
    value_parent = compute_node_value(sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization)
    si_parent = splitter.find_node_split(n_samples, hists_parent, sum_gradients, sum_hessians, value_parent)
    sample_indices_left, sample_indices_right, _ = splitter.split_indices(si_parent, sample_indices)
    hists_left = builder.compute_histograms_brute(sample_indices_left)
    value_left = compute_node_value(si_parent.sum_gradient_left, si_parent.sum_hessian_left, -np.inf, np.inf, l2_regularization)
    hists_right = builder.compute_histograms_brute(sample_indices_right)
    value_right = compute_node_value(si_parent.sum_gradient_right, si_parent.sum_hessian_right, -np.inf, np.inf, l2_regularization)
    si_left = splitter.find_node_split(n_samples, hists_left, si_parent.sum_gradient_left, si_parent.sum_hessian_left, value_left)
    si_right = splitter.find_node_split(n_samples, hists_right, si_parent.sum_gradient_right, si_parent.sum_hessian_right, value_right)
    for si, indices in ((si_parent, sample_indices), (si_left, sample_indices_left), (si_right, sample_indices_right)):
        gradient = si.sum_gradient_right + si.sum_gradient_left
        expected_gradient = all_gradients[indices].sum()
        hessian = si.sum_hessian_right + si.sum_hessian_left
        if constant_hessian:
            expected_hessian = indices.shape[0] * all_hessians[0]
        else:
            expected_hessian = all_hessians[indices].sum()
        assert np.isclose(gradient, expected_gradient)
        assert np.isclose(hessian, expected_hessian)
    hists_parent = np.asarray(hists_parent, dtype=HISTOGRAM_DTYPE)
    hists_left = np.asarray(hists_left, dtype=HISTOGRAM_DTYPE)
    hists_right = np.asarray(hists_right, dtype=HISTOGRAM_DTYPE)
    for hists, indices in ((hists_parent, sample_indices), (hists_left, sample_indices_left), (hists_right, sample_indices_right)):
        gradients = hists['sum_gradients'].sum(axis=1)
        expected_gradient = all_gradients[indices].sum()
        hessians = hists['sum_hessians'].sum(axis=1)
        if constant_hessian:
            expected_hessian = 0.0
        else:
            expected_hessian = all_hessians[indices].sum()
        assert np.allclose(gradients, expected_gradient)
        assert np.allclose(hessians, expected_hessian)