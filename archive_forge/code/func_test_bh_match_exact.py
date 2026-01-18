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
def test_bh_match_exact():
    random_state = check_random_state(0)
    n_features = 10
    X = random_state.randn(30, n_features).astype(np.float32)
    X_embeddeds = {}
    n_iter = {}
    for method in ['exact', 'barnes_hut']:
        tsne = TSNE(n_components=2, method=method, learning_rate=1.0, init='random', random_state=0, n_iter=251, perplexity=29.5, angle=0)
        tsne._EXPLORATION_N_ITER = 0
        X_embeddeds[method] = tsne.fit_transform(X)
        n_iter[method] = tsne.n_iter_
    assert n_iter['exact'] == n_iter['barnes_hut']
    assert_allclose(X_embeddeds['exact'], X_embeddeds['barnes_hut'], rtol=0.0001)