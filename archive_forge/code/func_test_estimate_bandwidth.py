import warnings
import numpy as np
import pytest
from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, mean_shift
from sklearn.datasets import make_blobs
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
def test_estimate_bandwidth():
    bandwidth = estimate_bandwidth(X, n_samples=200)
    assert 0.9 <= bandwidth <= 1.5