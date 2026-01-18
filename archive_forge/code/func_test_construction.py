import networkx as nx
def test_construction(self):
    G = nx.path_graph(2)
    M = nx.mycielskian(G)
    assert nx.is_isomorphic(M, nx.cycle_graph(5))