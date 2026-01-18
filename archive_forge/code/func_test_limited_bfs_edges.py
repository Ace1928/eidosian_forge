from functools import partial
import pytest
import networkx as nx
def test_limited_bfs_edges(self):
    edges = nx.bfs_edges(self.G, source=9, depth_limit=4)
    assert list(edges) == [(9, 8), (9, 10), (8, 7), (7, 2), (2, 1), (2, 3)]