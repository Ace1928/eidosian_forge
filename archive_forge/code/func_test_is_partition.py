import networkx as nx
def test_is_partition():
    G = nx.empty_graph(3)
    assert nx.community.is_partition(G, [{0, 1}, {2}])
    assert nx.community.is_partition(G, ({0, 1}, {2}))
    assert nx.community.is_partition(G, ([0, 1], [2]))
    assert nx.community.is_partition(G, [[0, 1], [2]])