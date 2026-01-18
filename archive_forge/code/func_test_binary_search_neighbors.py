import sys
from io import StringIO
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.optimize import check_grad
from scipy.spatial.distance import pdist, squareform
from sklearn import config_context
from sklearn.datasets import make_blobs
from sklearn.exceptions import EfficiencyWarning
from sklearn.manifold import (  # type: ignore
from sklearn.manifold._t_sne import (
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.metrics.pairwise import (
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
def test_binary_search_neighbors():
    n_samples = 200
    desired_perplexity = 25.0
    random_state = check_random_state(0)
    data = random_state.randn(n_samples, 2).astype(np.float32, copy=False)
    distances = pairwise_distances(data)
    P1 = _binary_search_perplexity(distances, desired_perplexity, verbose=0)
    n_neighbors = n_samples - 1
    nn = NearestNeighbors().fit(data)
    distance_graph = nn.kneighbors_graph(n_neighbors=n_neighbors, mode='distance')
    distances_nn = distance_graph.data.astype(np.float32, copy=False)
    distances_nn = distances_nn.reshape(n_samples, n_neighbors)
    P2 = _binary_search_perplexity(distances_nn, desired_perplexity, verbose=0)
    indptr = distance_graph.indptr
    P1_nn = np.array([P1[k, distance_graph.indices[indptr[k]:indptr[k + 1]]] for k in range(n_samples)])
    assert_array_almost_equal(P1_nn, P2, decimal=4)
    for k in np.linspace(150, n_samples - 1, 5):
        k = int(k)
        topn = k * 10
        distance_graph = nn.kneighbors_graph(n_neighbors=k, mode='distance')
        distances_nn = distance_graph.data.astype(np.float32, copy=False)
        distances_nn = distances_nn.reshape(n_samples, k)
        P2k = _binary_search_perplexity(distances_nn, desired_perplexity, verbose=0)
        assert_array_almost_equal(P1_nn, P2, decimal=2)
        idx = np.argsort(P1.ravel())[::-1]
        P1top = P1.ravel()[idx][:topn]
        idx = np.argsort(P2k.ravel())[::-1]
        P2top = P2k.ravel()[idx][:topn]
        assert_array_almost_equal(P1top, P2top, decimal=2)