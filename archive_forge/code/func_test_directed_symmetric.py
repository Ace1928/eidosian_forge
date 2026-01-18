import networkx as nx
def test_directed_symmetric(self):
    """Tests that a cut in a directed graph is symmetric."""
    G = nx.barbell_graph(3, 0).to_directed()
    S = {0, 1, 4}
    T = {2, 3, 5}
    assert nx.cut_size(G, S, T) == 8
    assert nx.cut_size(G, T, S) == 8