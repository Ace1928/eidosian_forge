import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_density_selfloop(self):
    G = nx.Graph()
    G.add_edge(1, 1)
    assert nx.density(G) == 0.0
    G.add_edge(1, 2)
    assert nx.density(G) == 2.0