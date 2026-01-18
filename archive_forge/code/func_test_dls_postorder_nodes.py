import networkx as nx
def test_dls_postorder_nodes(self):
    assert list(nx.dfs_postorder_nodes(self.G, source=3, depth_limit=3)) == [1, 7, 2, 5, 4, 3]
    assert list(nx.dfs_postorder_nodes(self.D, source=2, depth_limit=2)) == [3, 7, 2]