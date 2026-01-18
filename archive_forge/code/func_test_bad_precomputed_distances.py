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
@pytest.mark.parametrize('method, retype', [('exact', np.asarray), ('barnes_hut', np.asarray), *[('barnes_hut', csr_container) for csr_container in CSR_CONTAINERS]])
@pytest.mark.parametrize('D, message_regex', [([[0.0], [1.0]], '.* square distance matrix'), ([[0.0, -1.0], [1.0, 0.0]], '.* positive.*')])
def test_bad_precomputed_distances(method, D, retype, message_regex):
    tsne = TSNE(metric='precomputed', method=method, init='random', random_state=42, perplexity=1)
    with pytest.raises(ValueError, match=message_regex):
        tsne.fit_transform(retype(D))