from functools import partial
import pytest
import networkx as nx
def test_limited_bfs_tree(self):
    T = nx.bfs_tree(self.G, source=3, depth_limit=1)
    assert sorted(T.edges()) == [(3, 2), (3, 4)]