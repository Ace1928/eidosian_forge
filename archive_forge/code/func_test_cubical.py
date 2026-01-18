import pytest
import networkx as nx
def test_cubical(self):
    G = nx.cubical_graph()
    assert nx.generalized_degree(G, 0) == {0: 3}