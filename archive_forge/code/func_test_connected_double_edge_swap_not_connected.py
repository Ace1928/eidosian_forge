import pytest
import networkx as nx
def test_connected_double_edge_swap_not_connected():
    with pytest.raises(nx.NetworkXError):
        G = nx.path_graph(3)
        nx.add_path(G, [10, 11, 12])
        G = nx.connected_double_edge_swap(G)