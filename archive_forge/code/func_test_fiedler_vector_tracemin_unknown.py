from math import sqrt
import pytest
import networkx as nx
def test_fiedler_vector_tracemin_unknown():
    """Test that "tracemin_unknown" raises an exception."""
    pytest.importorskip('scipy')
    G = nx.barbell_graph(5, 4)
    L = nx.laplacian_matrix(G)
    X = np.asarray(np.random.normal(size=(1, L.shape[0]))).T
    with pytest.raises(nx.NetworkXError, match='Unknown linear system solver'):
        nx.linalg.algebraicconnectivity._tracemin_fiedler(L, X, normalized=False, tol=1e-08, method='tracemin_unknown')