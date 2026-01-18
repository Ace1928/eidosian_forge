import networkx as nx
def test_dls_labeled_edges_depth_1(self):
    edges = list(nx.dfs_labeled_edges(self.G, source=5, depth_limit=1))
    forward = [(u, v) for u, v, d in edges if d == 'forward']
    assert forward == [(5, 5), (5, 4), (5, 6)]
    assert edges == [(5, 5, 'forward'), (5, 4, 'forward'), (5, 4, 'reverse-depth_limit'), (5, 6, 'forward'), (5, 6, 'reverse-depth_limit'), (5, 5, 'reverse')]