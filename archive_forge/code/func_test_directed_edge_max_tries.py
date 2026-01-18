import pytest
import networkx as nx
def test_directed_edge_max_tries():
    with pytest.raises(nx.NetworkXAlgorithmError):
        G = nx.directed_edge_swap(nx.complete_graph(4, nx.DiGraph()), nswap=1, max_tries=5)