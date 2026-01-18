import pytest
import networkx as nx
def test_mu_too_small():
    with pytest.raises(nx.NetworkXError, match='mu must be in the interval \\[0, 1\\]'):
        n = 100
        tau1 = 2
        tau2 = 2
        mu = -1
        nx.LFR_benchmark_graph(n, tau1, tau2, mu, min_degree=2)