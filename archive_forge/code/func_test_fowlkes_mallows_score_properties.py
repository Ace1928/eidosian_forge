import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_fowlkes_mallows_score_properties():
    labels_a = np.array([0, 0, 0, 1, 1, 2])
    labels_b = np.array([1, 1, 2, 2, 0, 0])
    expected = 1.0 / np.sqrt((1.0 + 3.0) * (1.0 + 2.0))
    score_original = fowlkes_mallows_score(labels_a, labels_b)
    assert_almost_equal(score_original, expected)
    score_symmetric = fowlkes_mallows_score(labels_b, labels_a)
    assert_almost_equal(score_symmetric, expected)
    score_permuted = fowlkes_mallows_score((labels_a + 1) % 3, labels_b)
    assert_almost_equal(score_permuted, expected)
    score_both = fowlkes_mallows_score(labels_b, (labels_a + 2) % 3)
    assert_almost_equal(score_both, expected)