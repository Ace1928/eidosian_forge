import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_fowlkes_mallows_score():
    score = fowlkes_mallows_score([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2])
    assert_almost_equal(score, 4.0 / np.sqrt(12.0 * 6.0))
    perfect_score = fowlkes_mallows_score([0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0])
    assert_almost_equal(perfect_score, 1.0)
    worst_score = fowlkes_mallows_score([0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5])
    assert_almost_equal(worst_score, 0.0)