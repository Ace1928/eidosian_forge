import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
@pytest.mark.parametrize('dt', (None, int, np.dtype([('weight', 'f8'), ('color', 'i1')])))
def test_from_numpy_array_no_edge_attr(self, dt):
    A = np.array([[0, 1], [1, 0]], dtype=dt)
    G = nx.from_numpy_array(A, edge_attr=None)
    assert 'weight' not in G.edges[0, 1]
    assert len(G.edges[0, 1]) == 0