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
def test_accessible_kl_divergence():
    random_state = check_random_state(0)
    X = random_state.randn(50, 2)
    tsne = TSNE(n_iter_without_progress=2, verbose=2, random_state=0, method='exact', n_iter=500)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        tsne.fit_transform(X)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    for line in out.split('\n')[::-1]:
        if 'Iteration' in line:
            _, _, error = line.partition('error = ')
            if error:
                error, _, _ = error.partition(',')
                break
    assert_almost_equal(tsne.kl_divergence_, float(error), decimal=5)