import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_degree_centrality(self):
    d = bipartite.degree_centrality(self.P4, [1, 3])
    answer = {0: 0.5, 1: 1.0, 2: 1.0, 3: 0.5}
    assert d == answer
    d = bipartite.degree_centrality(self.K3, [0, 1, 2])
    answer = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
    assert d == answer
    d = bipartite.degree_centrality(self.C4, [0, 2])
    answer = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    assert d == answer