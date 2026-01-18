import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
def test_calinski_harabasz_score():
    assert_raises_on_only_one_label(calinski_harabasz_score)
    assert_raises_on_all_points_same_cluster(calinski_harabasz_score)
    assert 1.0 == calinski_harabasz_score(np.ones((10, 2)), [0] * 5 + [1] * 5)
    assert 0.0 == calinski_harabasz_score([[-1, -1], [1, 1]] * 10, [0] * 10 + [1] * 10)
    X = [[0, 0], [1, 1]] * 5 + [[3, 3], [4, 4]] * 5 + [[0, 4], [1, 3]] * 5 + [[3, 1], [4, 0]] * 5
    labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    pytest.approx(calinski_harabasz_score(X, labels), 45 * (40 - 4) / (5 * (4 - 1)))