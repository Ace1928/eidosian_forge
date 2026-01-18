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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_high_perplexity_precomputed_sparse_distances(csr_container):
    dist = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    bad_dist = csr_container(dist)
    tsne = TSNE(metric='precomputed', init='random', random_state=42, perplexity=1)
    msg = '3 neighbors per samples are required, but some samples have only 1'
    with pytest.raises(ValueError, match=msg):
        tsne.fit_transform(bad_dist)