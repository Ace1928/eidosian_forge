import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_bipartite_sets_disconnected(self):
    with pytest.raises(nx.AmbiguousSolution):
        G = nx.path_graph(4)
        G.add_edges_from([(5, 6), (6, 7)])
        X, Y = bipartite.sets(G)