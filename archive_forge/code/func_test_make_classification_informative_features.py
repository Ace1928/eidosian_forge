import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_classification_informative_features():
    """Test the construction of informative features in make_classification

    Also tests `n_clusters_per_class`, `n_classes`, `hypercube` and
    fully-specified `weights`.
    """
    class_sep = 1000000.0
    make = partial(make_classification, class_sep=class_sep, n_redundant=0, n_repeated=0, flip_y=0, shift=0, scale=1, shuffle=False)
    for n_informative, weights, n_clusters_per_class in [(2, [1], 1), (2, [1 / 3] * 3, 1), (2, [1 / 4] * 4, 1), (2, [1 / 2] * 2, 2), (2, [3 / 4, 1 / 4], 2), (10, [1 / 3] * 3, 10), (int(64), [1], 1)]:
        n_classes = len(weights)
        n_clusters = n_classes * n_clusters_per_class
        n_samples = n_clusters * 50
        for hypercube in (False, True):
            X, y = make(n_samples=n_samples, n_classes=n_classes, weights=weights, n_features=n_informative, n_informative=n_informative, n_clusters_per_class=n_clusters_per_class, hypercube=hypercube, random_state=0)
            assert X.shape == (n_samples, n_informative)
            assert y.shape == (n_samples,)
            signs = np.sign(X)
            signs = signs.view(dtype='|S{0}'.format(signs.strides[0])).ravel()
            unique_signs, cluster_index = np.unique(signs, return_inverse=True)
            assert len(unique_signs) == n_clusters, 'Wrong number of clusters, or not in distinct quadrants'
            clusters_by_class = defaultdict(set)
            for cluster, cls in zip(cluster_index, y):
                clusters_by_class[cls].add(cluster)
            for clusters in clusters_by_class.values():
                assert len(clusters) == n_clusters_per_class, 'Wrong number of clusters per class'
            assert len(clusters_by_class) == n_classes, 'Wrong number of classes'
            assert_array_almost_equal(np.bincount(y) / len(y) // weights, [1] * n_classes, err_msg='Wrong number of samples per class')
            for cluster in range(len(unique_signs)):
                centroid = X[cluster_index == cluster].mean(axis=0)
                if hypercube:
                    assert_array_almost_equal(np.abs(centroid) / class_sep, np.ones(n_informative), decimal=5, err_msg='Clusters are not centered on hypercube vertices')
                else:
                    with pytest.raises(AssertionError):
                        assert_array_almost_equal(np.abs(centroid) / class_sep, np.ones(n_informative), decimal=5, err_msg='Clusters should not be centered on hypercube vertices')
    with pytest.raises(ValueError):
        make(n_features=2, n_informative=2, n_classes=5, n_clusters_per_class=1)
    with pytest.raises(ValueError):
        make(n_features=2, n_informative=2, n_classes=3, n_clusters_per_class=2)