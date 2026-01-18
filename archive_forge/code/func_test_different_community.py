import math
from functools import partial
import pytest
import networkx as nx
def test_different_community(self):
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    G.nodes[0]['community'] = 0
    G.nodes[1]['community'] = 0
    G.nodes[2]['community'] = 0
    G.nodes[3]['community'] = 1
    self.test(G, [(0, 3)], [(0, 3, 0)])