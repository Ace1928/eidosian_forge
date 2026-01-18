import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_bipartite_sets_given_top_nodes(self):
    G = nx.path_graph(4)
    top_nodes = [0, 2]
    X, Y = bipartite.sets(G, top_nodes)
    assert X == {0, 2}
    assert Y == {1, 3}