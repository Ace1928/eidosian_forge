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
def test_binary_search_underflow():
    random_state = check_random_state(42)
    data = random_state.randn(1, 90).astype(np.float32) + 100
    desired_perplexity = 30.0
    P = _binary_search_perplexity(data, desired_perplexity, verbose=0)
    perplexity = 2 ** (-np.nansum(P[0, 1:] * np.log2(P[0, 1:])))
    assert_almost_equal(perplexity, desired_perplexity, decimal=3)