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
def test_min_grad_norm():
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    min_grad_norm = 0.002
    tsne = TSNE(min_grad_norm=min_grad_norm, verbose=2, random_state=0, method='exact')
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        tsne.fit_transform(X)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    lines_out = out.split('\n')
    gradient_norm_values = []
    for line in lines_out:
        if 'Finished' in line:
            break
        start_grad_norm = line.find('gradient norm')
        if start_grad_norm >= 0:
            line = line[start_grad_norm:]
            line = line.replace('gradient norm = ', '').split(' ')[0]
            gradient_norm_values.append(float(line))
    gradient_norm_values = np.array(gradient_norm_values)
    n_smaller_gradient_norms = len(gradient_norm_values[gradient_norm_values <= min_grad_norm])
    assert n_smaller_gradient_norms <= 1