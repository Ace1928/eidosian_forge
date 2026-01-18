import math
from functools import partial
import pytest
import networkx as nx
def test_insufficient_community_information(self):
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    G.nodes[0]['community'] = 0
    G.nodes[1]['community'] = 0
    G.nodes[3]['community'] = 0
    assert pytest.raises(nx.NetworkXAlgorithmError, list, self.func(G, [(0, 3)]))