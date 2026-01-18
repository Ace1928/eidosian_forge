import pytest
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.degree_seq import havel_hakimi_graph
def test_incidence_matrix_simple():
    deg = [3, 2, 2, 1, 0]
    G = havel_hakimi_graph(deg)
    deg = [(1, 0), (1, 0), (1, 0), (2, 0), (1, 0), (2, 1), (0, 1), (0, 1)]
    MG = nx.random_clustered_graph(deg, seed=42)
    I = nx.incidence_matrix(G, dtype=int).todense()
    expected = np.array([[1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]])
    np.testing.assert_equal(I, expected)
    I = nx.incidence_matrix(MG, dtype=int).todense()
    expected = np.array([[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 1]])
    np.testing.assert_equal(I, expected)
    with pytest.raises(NetworkXError):
        nx.incidence_matrix(G, nodelist=[0, 1])