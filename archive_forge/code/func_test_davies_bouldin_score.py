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
def test_davies_bouldin_score():
    assert_raises_on_only_one_label(davies_bouldin_score)
    assert_raises_on_all_points_same_cluster(davies_bouldin_score)
    assert davies_bouldin_score(np.ones((10, 2)), [0] * 5 + [1] * 5) == pytest.approx(0.0)
    assert davies_bouldin_score([[-1, -1], [1, 1]] * 10, [0] * 10 + [1] * 10) == pytest.approx(0.0)
    X = [[0, 0], [1, 1]] * 5 + [[3, 3], [4, 4]] * 5 + [[0, 4], [1, 3]] * 5 + [[3, 1], [4, 0]] * 5
    labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    pytest.approx(davies_bouldin_score(X, labels), 2 * np.sqrt(0.5) / 3)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        davies_bouldin_score(X, labels)
    X = [[0, 0], [2, 2], [3, 3], [5, 5]]
    labels = [0, 0, 1, 2]
    pytest.approx(davies_bouldin_score(X, labels), 5.0 / 4 / 3)