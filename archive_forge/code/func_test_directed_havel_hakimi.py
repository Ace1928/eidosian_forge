import pytest
import networkx as nx
def test_directed_havel_hakimi():
    n, r = (100, 10)
    p = 1.0 / r
    for i in range(r):
        G1 = nx.erdos_renyi_graph(n, p * (i + 1), None, True)
        din1 = [d for n, d in G1.in_degree()]
        dout1 = [d for n, d in G1.out_degree()]
        G2 = nx.directed_havel_hakimi_graph(din1, dout1)
        din2 = [d for n, d in G2.in_degree()]
        dout2 = [d for n, d in G2.out_degree()]
        assert sorted(din1) == sorted(din2)
        assert sorted(dout1) == sorted(dout2)
    dout = [1000, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    din = [103, 102, 102, 102, 102, 102, 102, 102, 102, 102]
    pytest.raises(nx.exception.NetworkXError, nx.directed_havel_hakimi_graph, din, dout)
    dout = [1, 1, 1, 1, 1, 2, 2, 2, 3, 4]
    din = [2, 2, 2, 2, 2, 2, 2, 2, 0, 2]
    G2 = nx.directed_havel_hakimi_graph(din, dout)
    dout2 = (d for n, d in G2.out_degree())
    din2 = (d for n, d in G2.in_degree())
    assert sorted(dout) == sorted(dout2)
    assert sorted(din) == sorted(din2)
    din = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    pytest.raises(nx.exception.NetworkXError, nx.directed_havel_hakimi_graph, din, dout)
    din = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, -2]
    pytest.raises(nx.exception.NetworkXError, nx.directed_havel_hakimi_graph, din, dout)