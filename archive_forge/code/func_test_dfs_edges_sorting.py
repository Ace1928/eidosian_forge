import networkx as nx
def test_dfs_edges_sorting(self):
    G = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 4), (3, 0), (0, 4)])
    edges_asc = nx.dfs_edges(G, source=0, sort_neighbors=sorted)
    sorted_desc = lambda x: sorted(x, reverse=True)
    edges_desc = nx.dfs_edges(G, source=0, sort_neighbors=sorted_desc)
    assert list(edges_asc) == [(0, 1), (1, 2), (2, 4), (1, 3)]
    assert list(edges_desc) == [(0, 4), (4, 2), (2, 1), (1, 3)]