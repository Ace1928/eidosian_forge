import math
from functools import partial
import pytest
import networkx as nx
def test_all_nonexistent_edges(self):
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (2, 3)])
    G.nodes[0]['community'] = 0
    G.nodes[1]['community'] = 1
    G.nodes[2]['community'] = 0
    G.nodes[3]['community'] = 0
    self.test(G, None, [(0, 3, 1 / self.delta), (1, 2, 0), (1, 3, 0)])