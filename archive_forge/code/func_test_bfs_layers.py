from functools import partial
import pytest
import networkx as nx
def test_bfs_layers(self):
    expected = {0: [0], 1: [1], 2: [2, 3], 3: [4]}
    assert dict(enumerate(nx.bfs_layers(self.G, sources=[0]))) == expected
    assert dict(enumerate(nx.bfs_layers(self.G, sources=0))) == expected