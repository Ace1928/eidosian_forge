import networkx as nx
def test_dls_preorder_nodes(self):
    assert list(nx.dfs_preorder_nodes(self.G, source=0, depth_limit=2)) == [0, 1, 2]
    assert list(nx.dfs_preorder_nodes(self.D, source=1, depth_limit=2)) == [1, 0]