import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS
def test_compute_sample_weight():
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight('balanced', y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    sample_weight = compute_sample_weight({1: 2, 2: 1}, y)
    assert_array_almost_equal(sample_weight, [2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    y = np.asarray([[1], [1], [1], [2], [2], [2]])
    sample_weight = compute_sample_weight('balanced', y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = np.asarray([1, 1, 1, 2, 2, 2, 3])
    sample_weight = compute_sample_weight('balanced', y)
    expected_balanced = np.array([0.7777, 0.7777, 0.7777, 0.7777, 0.7777, 0.7777, 2.3333])
    assert_array_almost_equal(sample_weight, expected_balanced, decimal=4)
    sample_weight = compute_sample_weight(None, y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    sample_weight = compute_sample_weight('balanced', y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    sample_weight = compute_sample_weight([{1: 2, 2: 1}, {0: 1, 1: 2}], y)
    assert_array_almost_equal(sample_weight, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1], [3, -1]])
    sample_weight = compute_sample_weight('balanced', y)
    assert_array_almost_equal(sample_weight, expected_balanced ** 2, decimal=3)