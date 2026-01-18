import pytest
import networkx as nx
@pytest.mark.parametrize('n', (2, 3, 5, 6, 10))
def test_margulis_gabber_galil_graph_eigvals(n):
    np = pytest.importorskip('numpy')
    sp = pytest.importorskip('scipy')
    g = nx.margulis_gabber_galil_graph(n)
    w = sorted(sp.linalg.eigvalsh(nx.adjacency_matrix(g).toarray()))
    assert w[-2] < 5 * np.sqrt(2)