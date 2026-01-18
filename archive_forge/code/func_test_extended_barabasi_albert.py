import pytest
import networkx as nx
def test_extended_barabasi_albert(self, m=2):
    """
        Tests that the extended BA random graph generated behaves consistently.

        Tests the exceptions are raised as expected.

        The graphs generation are repeated several times to prevent lucky-shots

        """
    seeds = [42, 314, 2718]
    for seed in seeds:
        BA_model = nx.barabasi_albert_graph(100, m, seed)
        BA_model_edges = BA_model.number_of_edges()
        G1 = nx.extended_barabasi_albert_graph(100, m, 0, 0, seed)
        assert G1.size() == BA_model_edges
        G1 = nx.extended_barabasi_albert_graph(100, m, 0.8, 0, seed)
        assert G1.size() > BA_model_edges * 2
        G2 = nx.extended_barabasi_albert_graph(100, m, 0, 0.8, seed)
        assert G2.size() == BA_model_edges
        G3 = nx.extended_barabasi_albert_graph(100, m, 0.3, 0.3, seed)
        assert G3.size() > G2.size()
        assert G3.size() < G1.size()
    ebag = nx.extended_barabasi_albert_graph
    pytest.raises(nx.NetworkXError, ebag, m, m, 0, 0)
    pytest.raises(nx.NetworkXError, ebag, 1, 0.5, 0, 0)
    pytest.raises(nx.NetworkXError, ebag, 100, 2, 0.5, 0.5)