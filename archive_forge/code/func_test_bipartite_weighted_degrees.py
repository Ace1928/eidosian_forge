import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_bipartite_weighted_degrees(self):
    G = nx.path_graph(5)
    G.add_edge(0, 1, weight=0.1, other=0.2)
    X = {1, 3}
    Y = {0, 2, 4}
    u, d = bipartite.degrees(G, Y, weight='weight')
    assert dict(u) == {1: 1.1, 3: 2}
    assert dict(d) == {0: 0.1, 2: 2, 4: 1}
    u, d = bipartite.degrees(G, Y, weight='other')
    assert dict(u) == {1: 1.2, 3: 2}
    assert dict(d) == {0: 0.2, 2: 2, 4: 1}