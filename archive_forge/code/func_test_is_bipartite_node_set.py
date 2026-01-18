import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_is_bipartite_node_set(self):
    G = nx.path_graph(4)
    with pytest.raises(nx.AmbiguousSolution):
        bipartite.is_bipartite_node_set(G, [1, 1, 2, 3])
    assert bipartite.is_bipartite_node_set(G, [0, 2])
    assert bipartite.is_bipartite_node_set(G, [1, 3])
    assert not bipartite.is_bipartite_node_set(G, [1, 2])
    G.add_edge(10, 20)
    assert bipartite.is_bipartite_node_set(G, [0, 2, 10])
    assert bipartite.is_bipartite_node_set(G, [0, 2, 20])
    assert bipartite.is_bipartite_node_set(G, [1, 3, 10])
    assert bipartite.is_bipartite_node_set(G, [1, 3, 20])