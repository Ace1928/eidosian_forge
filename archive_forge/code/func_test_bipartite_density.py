import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_bipartite_density(self):
    G = nx.path_graph(5)
    X, Y = bipartite.sets(G)
    density = len(list(G.edges())) / (len(X) * len(Y))
    assert bipartite.density(G, X) == density
    D = nx.DiGraph(G.edges())
    assert bipartite.density(D, X) == density / 2.0
    assert bipartite.density(nx.Graph(), {}) == 0.0