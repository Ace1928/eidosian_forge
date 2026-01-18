from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import manifold, neighbors
from sklearn.datasets import make_blobs
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph
from sklearn.utils._testing import (
def test_barycenter_kneighbors_graph(global_dtype):
    X = np.array([[0, 1], [1.01, 1.0], [2, 0]], dtype=global_dtype)
    graph = barycenter_kneighbors_graph(X, 1)
    expected_graph = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=global_dtype)
    assert graph.dtype == global_dtype
    assert_allclose(graph.toarray(), expected_graph)
    graph = barycenter_kneighbors_graph(X, 2)
    assert_allclose(np.sum(graph.toarray(), axis=1), np.ones(3))
    pred = np.dot(graph.toarray(), X)
    assert linalg.norm(pred - X) / X.shape[0] < 1