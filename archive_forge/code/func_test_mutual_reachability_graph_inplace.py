import numpy as np
import pytest
from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.utils._testing import (
@pytest.mark.parametrize('array_type', ['array', 'sparse_csr'])
def test_mutual_reachability_graph_inplace(array_type):
    """Check that the operation is happening inplace."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = X.T @ X
    np.fill_diagonal(X, 0.0)
    X = _convert_container(X, array_type)
    mr_graph = mutual_reachability_graph(X)
    assert id(mr_graph) == id(X)