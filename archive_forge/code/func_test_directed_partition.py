import networkx as nx
def test_directed_partition():
    """
    Test 2 cases that were looping infinitely
    from issues #5175 and #5704
    """
    G = nx.DiGraph()
    H = nx.DiGraph()
    G.add_nodes_from(range(10))
    H.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    G_edges = [(0, 2), (0, 1), (1, 0), (2, 1), (2, 0), (3, 4), (4, 3), (7, 8), (8, 7), (9, 10), (10, 9)]
    H_edges = [(1, 2), (1, 6), (1, 9), (2, 3), (2, 4), (2, 5), (3, 4), (4, 3), (4, 5), (5, 4), (6, 7), (6, 8), (9, 10), (9, 11), (10, 11), (11, 10)]
    G.add_edges_from(G_edges)
    H.add_edges_from(H_edges)
    G_expected_partition = [{0, 1, 2}, {3, 4}, {5}, {6}, {8, 7}, {9, 10}]
    G_partition = nx.community.louvain_communities(G, seed=123, weight=None)
    H_expected_partition = [{2, 3, 4, 5}, {8, 1, 6, 7}, {9, 10, 11}]
    H_partition = nx.community.louvain_communities(H, seed=123, weight=None)
    assert G_partition == G_expected_partition
    assert H_partition == H_expected_partition