import pytest
import networkx as nx
def test_invalid_tau2():
    with pytest.raises(nx.NetworkXError, match='tau1 must be greater than one'):
        n = 100
        tau1 = 1
        tau2 = 2
        mu = 0.1
        nx.LFR_benchmark_graph(n, tau1, tau2, mu, min_degree=2)