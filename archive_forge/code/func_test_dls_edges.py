import networkx as nx
def test_dls_edges(self):
    edges = nx.dfs_edges(self.G, source=9, depth_limit=4)
    assert list(edges) == [(9, 8), (8, 7), (7, 2), (2, 1), (2, 3), (9, 10)]