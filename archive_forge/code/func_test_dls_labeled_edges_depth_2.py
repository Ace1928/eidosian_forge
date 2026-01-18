import networkx as nx
def test_dls_labeled_edges_depth_2(self):
    edges = list(nx.dfs_labeled_edges(self.G, source=6, depth_limit=2))
    forward = [(u, v) for u, v, d in edges if d == 'forward']
    assert forward == [(6, 6), (6, 5), (5, 4)]
    assert edges == [(6, 6, 'forward'), (6, 5, 'forward'), (5, 4, 'forward'), (5, 4, 'reverse-depth_limit'), (5, 6, 'nontree'), (6, 5, 'reverse'), (6, 6, 'reverse')]