import pytest
import networkx as nx
def test_edge_betweenness_partition():
    G = nx.barbell_graph(3, 0)
    C = nx.community.edge_betweenness_partition(G, 2)
    answer = [{0, 1, 2}, {3, 4, 5}]
    assert len(C) == len(answer)
    for s in answer:
        assert s in C
    G = nx.barbell_graph(3, 1)
    C = nx.community.edge_betweenness_partition(G, 3)
    answer = [{0, 1, 2}, {4, 5, 6}, {3}]
    assert len(C) == len(answer)
    for s in answer:
        assert s in C
    C = nx.community.edge_betweenness_partition(G, 7)
    answer = [{n} for n in G]
    assert len(C) == len(answer)
    for s in answer:
        assert s in C
    C = nx.community.edge_betweenness_partition(G, 1)
    assert C == [set(G)]
    C = nx.community.edge_betweenness_partition(G, 1, weight='weight')
    assert C == [set(G)]
    with pytest.raises(nx.NetworkXError):
        nx.community.edge_betweenness_partition(G, 0)
    with pytest.raises(nx.NetworkXError):
        nx.community.edge_betweenness_partition(G, -1)
    with pytest.raises(nx.NetworkXError):
        nx.community.edge_betweenness_partition(G, 10)