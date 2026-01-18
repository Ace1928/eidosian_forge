from functools import partial
import pytest
import networkx as nx
def test_bfs_tree(self):
    T = nx.bfs_tree(self.G, source=0)
    assert sorted(T.nodes()) == sorted(self.G.nodes())
    assert sorted(T.edges()) == [(0, 1), (1, 2), (1, 3), (2, 4)]