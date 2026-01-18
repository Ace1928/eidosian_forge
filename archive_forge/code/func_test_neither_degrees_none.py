import pytest
import networkx as nx
def test_neither_degrees_none():
    with pytest.raises(nx.NetworkXError, match='Must assign exactly one of min_degree and average_degree'):
        n = 100
        tau1 = 2
        tau2 = 2
        mu = 1
        nx.LFR_benchmark_graph(n, tau1, tau2, mu, min_degree=2, average_degree=5)