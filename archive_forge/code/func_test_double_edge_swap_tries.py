import pytest
import networkx as nx
def test_double_edge_swap_tries():
    with pytest.raises(nx.NetworkXError):
        G = nx.double_edge_swap(nx.path_graph(10), nswap=1, max_tries=0)