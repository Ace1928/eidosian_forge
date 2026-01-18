import math
from itertools import product
import numpy as np
import pytest
from scipy.sparse import rand as sparse_rand
from sklearn import clone, datasets, manifold, neighbors, pipeline, preprocessing
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('metric, p, is_euclidean', [('euclidean', 2, True), ('manhattan', 1, False), ('minkowski', 1, False), ('minkowski', 2, True), (lambda x1, x2: np.sqrt(np.sum(x1 ** 2 + x2 ** 2)), 2, False)])
def test_different_metric(global_dtype, metric, p, is_euclidean):
    X, _ = datasets.make_blobs(random_state=0)
    X = X.astype(global_dtype, copy=False)
    reference = manifold.Isomap().fit_transform(X)
    embedding = manifold.Isomap(metric=metric, p=p).fit_transform(X)
    if is_euclidean:
        assert_allclose(embedding, reference)
    else:
        with pytest.raises(AssertionError, match='Not equal to tolerance'):
            assert_allclose(embedding, reference)