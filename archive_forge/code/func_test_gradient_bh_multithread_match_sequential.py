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
def test_gradient_bh_multithread_match_sequential():
    n_features = 10
    n_samples = 30
    n_components = 2
    degrees_of_freedom = 1
    angle = 3
    perplexity = 5
    random_state = check_random_state(0)
    data = random_state.randn(n_samples, n_features).astype(np.float32)
    params = random_state.randn(n_samples, n_components)
    n_neighbors = n_samples - 1
    distances_csr = NearestNeighbors().fit(data).kneighbors_graph(n_neighbors=n_neighbors, mode='distance')
    P_bh = _joint_probabilities_nn(distances_csr, perplexity, verbose=0)
    kl_sequential, grad_sequential = _kl_divergence_bh(params, P_bh, degrees_of_freedom, n_samples, n_components, angle=angle, skip_num_points=0, verbose=0, num_threads=1)
    for num_threads in [2, 4]:
        kl_multithread, grad_multithread = _kl_divergence_bh(params, P_bh, degrees_of_freedom, n_samples, n_components, angle=angle, skip_num_points=0, verbose=0, num_threads=num_threads)
        assert_allclose(kl_multithread, kl_sequential, rtol=1e-06)
        assert_allclose(grad_multithread, grad_multithread)