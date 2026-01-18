import pytest
import networkx as nx
def test_double_edge_swap_small():
    with pytest.raises(nx.NetworkXError):
        G = nx.double_edge_swap(nx.path_graph(3))