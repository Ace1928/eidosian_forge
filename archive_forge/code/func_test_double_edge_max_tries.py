import pytest
import networkx as nx
def test_double_edge_max_tries():
    with pytest.raises(nx.NetworkXAlgorithmError):
        G = nx.double_edge_swap(nx.complete_graph(4), nswap=1, max_tries=5)