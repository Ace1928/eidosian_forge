import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_from_numpy_array_custom_edge_attr(self):
    A = np.array([[0, 2], [3, 0]])
    G = nx.from_numpy_array(A, edge_attr='cost')
    assert 'weight' not in G.edges[0, 1]
    assert G.edges[0, 1]['cost'] == 3