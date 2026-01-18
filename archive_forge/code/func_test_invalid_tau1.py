import pytest
import networkx as nx
def test_invalid_tau1():
    with pytest.raises(nx.NetworkXError, match='tau2 must be greater than one'):
        n = 100
        tau1 = 2
        tau2 = 1
        mu = 0.1
        nx.LFR_benchmark_graph(n, tau1, tau2, mu, min_degree=2)