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
def test_early_exaggeration_used():
    random_state = check_random_state(0)
    n_components = 2
    methods = ['exact', 'barnes_hut']
    X = random_state.randn(25, n_components).astype(np.float32)
    for method in methods:
        tsne = TSNE(n_components=n_components, perplexity=1, learning_rate=100.0, init='pca', random_state=0, method=method, early_exaggeration=1.0, n_iter=250)
        X_embedded1 = tsne.fit_transform(X)
        tsne = TSNE(n_components=n_components, perplexity=1, learning_rate=100.0, init='pca', random_state=0, method=method, early_exaggeration=10.0, n_iter=250)
        X_embedded2 = tsne.fit_transform(X)
        assert not np.allclose(X_embedded1, X_embedded2)