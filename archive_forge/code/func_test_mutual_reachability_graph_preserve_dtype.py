import numpy as np
import pytest
from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.utils._testing import (
@pytest.mark.parametrize('array_type', ['array', 'sparse_csr'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_mutual_reachability_graph_preserve_dtype(array_type, dtype):
    """Check that the computation preserve dtype thanks to fused types."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = (X.T @ X).astype(dtype)
    np.fill_diagonal(X, 0.0)
    X = _convert_container(X, array_type)
    assert X.dtype == dtype
    mr_graph = mutual_reachability_graph(X)
    assert mr_graph.dtype == dtype