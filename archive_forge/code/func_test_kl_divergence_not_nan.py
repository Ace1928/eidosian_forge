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
@pytest.mark.parametrize('method', ['barnes_hut', 'exact'])
def test_kl_divergence_not_nan(method):
    random_state = check_random_state(0)
    X = random_state.randn(50, 2)
    tsne = TSNE(n_components=2, perplexity=2, learning_rate=100.0, random_state=0, method=method, verbose=0, n_iter=503, init='random')
    tsne.fit_transform(X)
    assert not np.isnan(tsne.kl_divergence_)