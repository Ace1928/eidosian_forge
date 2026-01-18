import networkx as nx
def test_dfs_labeled_disconnected_edges(self):
    edges = list(nx.dfs_labeled_edges(self.D))
    forward = [(u, v) for u, v, d in edges if d == 'forward']
    assert forward == [(0, 0), (0, 1), (2, 2), (2, 3)]
    assert edges == [(0, 0, 'forward'), (0, 1, 'forward'), (1, 0, 'nontree'), (0, 1, 'reverse'), (0, 0, 'reverse'), (2, 2, 'forward'), (2, 3, 'forward'), (3, 2, 'nontree'), (2, 3, 'reverse'), (2, 2, 'reverse')]