import networkx as nx
def test_is_isolate():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_node(2)
    assert not nx.is_isolate(G, 0)
    assert not nx.is_isolate(G, 1)
    assert nx.is_isolate(G, 2)