import networkx as nx
def test_dls_predecessor(self):
    assert nx.dfs_predecessors(self.G, source=0, depth_limit=3) == {1: 0, 2: 1, 3: 2, 7: 2}
    assert nx.dfs_predecessors(self.D, source=2, depth_limit=3) == {8: 7, 9: 8, 3: 2, 7: 2}