import networkx as nx
def test_dfs_labeled_edges_sorting(self):
    G = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 4), (3, 0), (0, 4)])
    edges_asc = nx.dfs_labeled_edges(G, source=0, sort_neighbors=sorted)
    sorted_desc = lambda x: sorted(x, reverse=True)
    edges_desc = nx.dfs_labeled_edges(G, source=0, sort_neighbors=sorted_desc)
    assert list(edges_asc) == [(0, 0, 'forward'), (0, 1, 'forward'), (1, 0, 'nontree'), (1, 2, 'forward'), (2, 1, 'nontree'), (2, 4, 'forward'), (4, 0, 'nontree'), (4, 2, 'nontree'), (2, 4, 'reverse'), (1, 2, 'reverse'), (1, 3, 'forward'), (3, 0, 'nontree'), (3, 1, 'nontree'), (1, 3, 'reverse'), (0, 1, 'reverse'), (0, 3, 'nontree'), (0, 4, 'nontree'), (0, 0, 'reverse')]
    assert list(edges_desc) == [(0, 0, 'forward'), (0, 4, 'forward'), (4, 2, 'forward'), (2, 4, 'nontree'), (2, 1, 'forward'), (1, 3, 'forward'), (3, 1, 'nontree'), (3, 0, 'nontree'), (1, 3, 'reverse'), (1, 2, 'nontree'), (1, 0, 'nontree'), (2, 1, 'reverse'), (4, 2, 'reverse'), (4, 0, 'nontree'), (0, 4, 'reverse'), (0, 3, 'nontree'), (0, 1, 'nontree'), (0, 0, 'reverse')]