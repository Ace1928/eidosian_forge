import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_bipartite_directed(self):
    G = bipartite.random_graph(10, 10, 0.1, directed=True)
    assert bipartite.is_bipartite(G)