import networkx as nx
def test_narrow(self):
    """Tests that a narrow beam width may cause an incomplete search."""
    G = nx.cycle_graph(4)
    edges = nx.bfs_beam_edges(G, 0, identity, width=1)
    assert list(edges) == [(0, 3), (3, 2)]