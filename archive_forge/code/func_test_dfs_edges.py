import networkx as nx
def test_dfs_edges(self):
    edges = nx.dfs_edges(self.G, source=0)
    assert list(edges) == [(0, 1), (1, 2), (2, 4), (1, 3)]
    edges = nx.dfs_edges(self.D)
    assert list(edges) == [(0, 1), (2, 3)]